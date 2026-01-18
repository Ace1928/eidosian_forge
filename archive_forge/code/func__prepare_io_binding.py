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
def _prepare_io_binding(self, model: ort.InferenceSession, *model_inputs: torch.Tensor, ordered_input_names: List[str], known_output_shapes: Optional[Dict[str, Tuple[int]]]=None, outputs_to_not_bind: Optional[Union[Set[str], str]]=None) -> Tuple[ort.IOBinding, Dict[str, Tuple[int]], Dict[str, torch.Tensor]]:
    """
        Prepares IO binding for ONNX Runtime.

        Args:
            model (`ort.InferenceSession`):
                The model for which we want to bind the inputs and outputs.
            *model_inputs:
                The inputs of the model.
            ordered_input_names (`List[str]`):
                Names of the inputs, that must match with the order of model_inputs.
            known_output_shapes (`Optional[Dict[str, Tuple[int]]]`, defaults to `None`):
                It can be hard to infer all the output shapes from the inputs only. For instance for the past key /
                values. It is possible to explicitely pass the shape via this argument.
            outputs_to_not_bind (`Optional[Union[Set[str], str]]`, defaults to `None`):
                The names of the outputs that should not be bound.

        Returns:
            `Tuple[ort.IOBinding, Dict[str, Tuple[int]], Dict[str, torch.Tensor]`: The IOBinding object, a dictionary
            containing the shape of each output, and another one pointing to the buffers containing the outputs data.
        """
    io_binding = model.io_binding()
    name_to_np_type = TypeHelper.get_io_numpy_type_map(model)
    input_name_to_shape = {}
    for idx, tensor in enumerate(model_inputs):
        if tensor is None:
            continue
        name = ordered_input_names[idx]
        tensor = tensor.contiguous()
        input_name_to_shape[name] = tensor.shape
        data_ptr = tensor.data_ptr()
        if 'past' in name and data_ptr == 0:
            data_ptr = model_inputs[0].data_ptr()
        io_binding.bind_input(name, tensor.device.type, IOBindingHelper.get_device_index(self.device), name_to_np_type[name], tuple(tensor.shape), data_ptr)
    dimensions = {}
    for input_ in model.get_inputs():
        shape = input_.shape
        for idx, axis in enumerate(shape):
            if isinstance(axis, str):
                dimensions[axis] = input_name_to_shape[input_.name][idx]
    output_shapes = {}
    output_buffers = {}
    if known_output_shapes is None:
        known_output_shapes = {}
    if outputs_to_not_bind is None:
        outputs_to_not_bind = set()
    elif isinstance(outputs_to_not_bind, str):
        outputs_to_not_bind = {outputs_to_not_bind}
    for output_node in model.get_outputs():
        output_name = output_node.name
        if output_name in outputs_to_not_bind:
            continue
        if output_name in known_output_shapes:
            output_shape = known_output_shapes[output_name]
        else:
            output_shape = []
            for axis_name in output_node.shape:
                output_shape.append(self._output_shape_inference(axis_name, dimensions))
        output_buffer = self._prepare_output_buffer(model, output_shape, output_name)
        io_binding.bind_output(output_name, output_buffer.device.type, IOBindingHelper.get_device_index(self.device), name_to_np_type[output_name], output_shape, output_buffer.data_ptr())
        output_shapes[output_name] = output_shape
        output_buffers[output_name] = output_buffer
    return (io_binding, output_shapes, output_buffers)