from __future__ import annotations
import logging
import os
from typing import Callable, Optional, Union
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors import SafetensorError, safe_open
from transformers.utils import cached_file
from transformers.utils.hub import get_checkpoint_shard_files
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
class _SafetensorLoader:
    """
    Simple utility class that loads tensors with safetensors from a single file or sharded files.

    Takes care of file name normalization etc.

    """

    def __init__(self, peft_model, model_path):
        if model_path is None:
            try:
                model_path = snapshot_download(peft_model.base_model.config._name_or_path, local_files_only=True)
            except AttributeError as exc:
                raise ValueError('The provided model does not appear to be a transformers model. In this case, you must pass the model_path to the safetensors file.') from exc
            except LocalEntryNotFoundError as exc:
                raise ValueError('The model.safetensors file must be present on disk, but it could not be found.') from exc
        suffix = 'model.safetensors'
        if not model_path.endswith(suffix):
            model_path = os.path.join(model_path, suffix)
        self.model_path = model_path
        self.base_model_prefix = getattr(peft_model.get_base_model(), 'base_model_prefix', None)
        self.prefix = 'base_model.model.'
        self.is_sharded = False
        self.weight_map = None
        if not os.path.exists(model_path):
            par_dir = model_path.rpartition(os.path.sep)[0]
            try:
                resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(par_dir, cached_file(par_dir, 'model.safetensors.index.json'))
            except OSError as exc:
                raise FileNotFoundError(f'Could not find file for {model_path}, ensure that there is a (sharded) safetensors file of the model.') from exc
            self.is_sharded = True
            file_map = {k.rpartition(os.path.sep)[-1]: k for k in resolved_archive_file}
            self.weight_map = {k: file_map[v] for k, v in sharded_metadata['weight_map'].items()}

    def get_tensor(self, name):
        if not self.is_sharded:
            file_path = self.model_path
        else:
            file_path = self.weight_map[name]
        with safe_open(file_path, framework='pt', device='cpu') as f:
            try:
                tensor = f.get_tensor(name)
            except SafetensorError as exc:
                if self.base_model_prefix:
                    name = name[len(self.base_model_prefix) + 1:]
                    tensor = f.get_tensor(name)
                else:
                    raise exc
        return tensor