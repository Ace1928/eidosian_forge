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
def _prepare_output_buffer(self, model: ort.InferenceSession, output_shape: Tuple[int], output_name: str):
    """Prepares the buffer of output_name with a 1D tensor."""
    ort_type = TypeHelper.get_output_type(model, output_name)
    torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
    if len(output_shape) > 0:
        output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self.device).contiguous()
    else:
        output_buffer = torch.tensor(0, dtype=torch_type, device=self.device).contiguous()
    return output_buffer