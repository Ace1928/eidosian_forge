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
class ORTModelForCustomTasks(ORTModel):
    """
    ONNX Model for any custom tasks. It can be used to leverage the inference acceleration for any single-file ONNX model, that may use custom inputs and outputs.
    """

    @add_start_docstrings_to_model_forward(CUSTOM_TASKS_EXAMPLE.format(processor_class=_TOKENIZER_FOR_DOC, model_class='ORTModelForCustomTasks', checkpoint='optimum/sbert-all-MiniLM-L6-with-pooler'))
    def forward(self, **kwargs):
        use_torch = isinstance(next(iter(kwargs.values())), torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == 'cuda' and self.use_io_binding:
            io_binding = IOBindingHelper.prepare_io_binding(self, **kwargs, ordered_input_names=self._ordered_input_names)
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            outputs = {}
            for name, output in zip(self.output_names.keys(), io_binding._iobinding.get_outputs()):
                outputs[name] = IOBindingHelper.to_pytorch(output)
            return ModelOutput(**outputs)
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch=use_torch, **kwargs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            outputs = self._prepare_onnx_outputs(onnx_outputs, use_torch=use_torch)
            return ModelOutput(outputs)

    def _prepare_onnx_inputs(self, use_torch: bool, **kwargs):
        onnx_inputs = {}
        for input in self.inputs_names.keys():
            onnx_inputs[input] = kwargs.pop(input)
            if use_torch:
                onnx_inputs[input] = onnx_inputs[input].cpu().detach().numpy()
        return onnx_inputs

    def _prepare_onnx_outputs(self, onnx_outputs, use_torch: bool):
        outputs = {}
        for output, idx in self.output_names.items():
            outputs[output] = onnx_outputs[idx]
            if use_torch:
                outputs[output] = torch.from_numpy(outputs[output]).to(self.device)
        return outputs