import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import (
from transformers.file_utils import add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
import onnxruntime as ort
from ..exporters import TasksManager
from ..exporters.onnx import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..onnx.utils import _get_external_data_paths
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .io_binding import IOBindingHelper, TypeHelper
from .utils import (
@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForFeatureExtraction(ORTModel):
    """
    ONNX Model for feature-extraction task.
    """
    auto_model_class = AutoModel

    @add_start_docstrings_to_model_forward(ONNX_TEXT_INPUTS_DOCSTRING.format('batch_size, sequence_length') + FEATURE_EXTRACTION_EXAMPLE.format(processor_class=_TOKENIZER_FOR_DOC, model_class='ORTModelForFeatureExtraction', checkpoint='optimum/all-MiniLM-L6-v2'))
    def forward(self, input_ids: Optional[Union[torch.Tensor, np.ndarray]]=None, attention_mask: Optional[Union[torch.Tensor, np.ndarray]]=None, token_type_ids: Optional[Union[torch.Tensor, np.ndarray]]=None, **kwargs):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == 'cuda' and self.use_io_binding:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_ids, attention_mask, token_type_ids, ordered_input_names=self._ordered_input_names)
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            return BaseModelOutput(last_hidden_state=output_buffers['last_hidden_state'].view(output_shapes['last_hidden_state']))
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                if attention_mask is None:
                    attention_mask = np.ones_like(input_ids)
                else:
                    attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()
            onnx_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if token_type_ids is not None:
                onnx_inputs['token_type_ids'] = token_type_ids
            outputs = self.model.run(None, onnx_inputs)
            last_hidden_state = outputs[self.output_names['last_hidden_state']]
            if use_torch:
                last_hidden_state = torch.from_numpy(last_hidden_state).to(self.device)
            return BaseModelOutput(last_hidden_state=last_hidden_state)

    @classmethod
    def _export(cls, model_id: str, config: 'PretrainedConfig', use_auth_token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, force_download: bool=False, cache_dir: Optional[str]=None, subfolder: str='', local_files_only: bool=False, trust_remote_code: bool=False, provider: str='CPUExecutionProvider', session_options: Optional[ort.SessionOptions]=None, provider_options: Optional[Dict[str, Any]]=None, use_io_binding: Optional[bool]=None, task: Optional[str]=None) -> 'ORTModel':
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        main_export(model_name_or_path=model_id, output=save_dir_path, task=task, do_validation=False, no_post_process=True, subfolder=subfolder, revision=revision, cache_dir=cache_dir, use_auth_token=use_auth_token, local_files_only=local_files_only, force_download=force_download, trust_remote_code=trust_remote_code, library_name='transformers')
        config.save_pretrained(save_dir_path)
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)
        return cls._from_pretrained(save_dir_path, config, use_io_binding=use_io_binding, model_save_dir=save_dir, provider=provider, session_options=session_options, provider_options=provider_options)