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
class ORTModelForMultipleChoice(ORTModel):
    """
    ONNX Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks. This class officially supports albert, bert, camembert, convbert, data2vec_text, deberta_v2, distilbert, electra, flaubert, ibert, mobilebert, nystromformer, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """
    auto_model_class = AutoModelForMultipleChoice

    @add_start_docstrings_to_model_forward(ONNX_TEXT_INPUTS_DOCSTRING.format('batch_size, sequence_length') + MULTIPLE_CHOICE_EXAMPLE.format(processor_class=_TOKENIZER_FOR_DOC, model_class='ORTModelForMultipleChoice', checkpoint='ehdwns1516/bert-base-uncased_SWAG'))
    def forward(self, input_ids: Optional[Union[torch.Tensor, np.ndarray]]=None, attention_mask: Optional[Union[torch.Tensor, np.ndarray]]=None, token_type_ids: Optional[Union[torch.Tensor, np.ndarray]]=None, **kwargs):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == 'cuda' and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_ids, attention_mask, token_type_ids, ordered_input_names=self._ordered_input_names)
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            return MultipleChoiceModelOutput(logits=output_buffers['logits'].view(output_shapes['logits']))
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()
            onnx_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if token_type_ids is not None:
                onnx_inputs['token_type_ids'] = token_type_ids
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names['logits']]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
            return MultipleChoiceModelOutput(logits=logits)