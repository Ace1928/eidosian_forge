import json
import logging
import os
from copy import deepcopy
from typing import Optional
import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
from safetensors.torch import load_file as safe_load_file
from transformers import PreTrainedModel
from ..import_utils import is_npu_available, is_peft_available, is_transformers_greater_than, is_xpu_available
@classmethod
def _get_current_device(cls):
    """
        Get the current device. For GPU, we return the local process index using the `accelerate.PartialState`
        object to handle corner cases when running scripts in distributed environments.

        Returns:
            current_device (`Union[int, str]`):
                The current device.
        """
    state = PartialState()
    if is_xpu_available():
        return f'xpu:{state.local_process_index}'
    elif is_npu_available():
        return f'npu:{state.local_process_index}'
    else:
        return state.local_process_index if torch.cuda.is_available() else 'cpu'