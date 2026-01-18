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
def _get_checkpoint_from_hub(cls, pretrained_model, pretrained_model_name_or_path, index_filename, token=None, model_name='pytorch_model.bin', model_index_name='pytorch_model.bin.index.json'):
    files_to_download = None
    filename = None
    is_resuming_training = True
    is_sharded = False
    try:
        filename = hf_hub_download(pretrained_model_name_or_path, model_name, token=token)
    except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
        if os.path.exists(index_filename):
            index_file_name = index_filename
        else:
            try:
                index_file_name = hf_hub_download(pretrained_model_name_or_path, model_index_name, token=token)
            except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                is_resuming_training = False
                logging.warning(f"A {type(pretrained_model)} model is loaded from '{pretrained_model_name_or_path}', and no v_head weight is found. This IS expected if you are not resuming PPO training.")
        if is_resuming_training:
            with open(index_file_name) as f:
                index = json.load(f)
            files_to_download = set()
            for k, v in index['weight_map'].items():
                if any((module in k for module in cls.supported_modules)):
                    files_to_download.add(v)
            is_sharded = True
    return (filename, files_to_download, is_sharded, is_resuming_training)