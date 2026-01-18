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
class ORTModelForCTC(ORTModel):
    """
    ONNX Model with a language modeling head on top for Connectionist Temporal Classification (CTC). This class officially supports data2vec_audio, hubert, sew, sew_d, unispeech, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """
    auto_model_class = AutoModelForCTC

    @add_start_docstrings_to_model_forward(ONNX_AUDIO_INPUTS_DOCSTRING.format('batch_size, sequence_length') + CTC_EXAMPLE.format(processor_class=_PROCESSOR_FOR_DOC, model_class='ORTModelForCTC', checkpoint='optimum/hubert-large-ls960-ft'))
    def forward(self, input_values: Optional[torch.Tensor]=None, **kwargs):
        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == 'cuda' and self.use_io_binding:
            input_size = input_values.shape[1]
            output_sizes = []

            def _conv_output_size(input_size, kernel_size, stride):
                return (input_size - kernel_size) // stride + 1
            for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
                input_size = _conv_output_size(input_size, kernel_size, stride)
                output_sizes.append(input_size)
            known_output_shapes = {'logits': [input_values.shape[0], output_sizes[-1], self.config.vocab_size]}
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_values, ordered_input_names=self._ordered_input_names, known_output_shapes=known_output_shapes)
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            outputs = {}
            return CausalLMOutput(logits=output_buffers['logits'].view(output_shapes['logits']))
        else:
            if use_torch:
                onnx_inputs = {'input_values': input_values.cpu().detach().numpy()}
            else:
                onnx_inputs = {'input_values': input_values}
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names['logits']]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
            return CausalLMOutput(logits=logits)