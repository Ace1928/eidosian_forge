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
def _split_kwargs(cls, kwargs):
    """
        Separate the kwargs from the arguments that we support inside
        `supported_args` and the ones that we don't.
        """
    check_peft_kwargs = False
    if is_peft_available():
        from peft import prepare_model_for_kbit_training
        check_peft_kwargs = True
    supported_kwargs = {}
    unsupported_kwargs = {}
    peft_kwargs = {}
    for key, value in kwargs.items():
        if key in cls.supported_args:
            supported_kwargs[key] = value
        else:
            unsupported_kwargs[key] = value
        if check_peft_kwargs:
            if key in prepare_model_for_kbit_training.__code__.co_varnames:
                peft_kwargs[key] = value
                if key in unsupported_kwargs:
                    unsupported_kwargs.pop(key)
    return (supported_kwargs, unsupported_kwargs, peft_kwargs)