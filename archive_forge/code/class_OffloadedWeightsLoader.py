import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from safetensors import safe_open
class OffloadedWeightsLoader(Mapping):
    """
    A collection that loads weights stored in a given state dict or memory-mapped on disk.

    Args:
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            A dictionary parameter name to tensor.
        save_folder (`str` or `os.PathLike`, *optional*):
            The directory in which the weights are stored (by `offload_state_dict` for instance).
        index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype`/ `shape` or safetensors filename). Will default
            to the index saved in `save_folder`.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]=None, save_folder: Optional[Union[str, os.PathLike]]=None, index: Mapping=None, device=None):
        if state_dict is None and save_folder is None and (index is None):
            raise ValueError('Need either a `state_dict`, a `save_folder` or an `index` containing offloaded weights.')
        self.state_dict = {} if state_dict is None else state_dict
        self.save_folder = save_folder
        if index is None and save_folder is not None:
            with open(os.path.join(save_folder, 'index.json')) as f:
                index = json.load(f)
        self.index = {} if index is None else index
        self.all_keys = list(self.state_dict.keys())
        self.all_keys.extend([key for key in self.index if key not in self.all_keys])
        self.device = device

    def __getitem__(self, key: str):
        if key in self.state_dict:
            return self.state_dict[key]
        weight_info = self.index[key]
        if weight_info.get('safetensors_file') is not None:
            device = 'cpu' if self.device is None else self.device
            tensor = None
            try:
                with safe_open(weight_info['safetensors_file'], framework='pt', device=device) as f:
                    tensor = f.get_tensor(weight_info.get('weight_name', key))
            except TypeError:
                with safe_open(weight_info['safetensors_file'], framework='pt', device='cpu') as f:
                    tensor = f.get_tensor(weight_info.get('weight_name', key))
            if 'dtype' in weight_info:
                tensor = tensor.to(getattr(torch, weight_info['dtype']))
            if tensor.device != torch.device(device):
                tensor = tensor.to(device)
            return tensor
        weight_file = os.path.join(self.save_folder, f'{key}.dat')
        return load_offloaded_weight(weight_file, weight_info)

    def __iter__(self):
        return iter(self.all_keys)

    def __len__(self):
        return len(self.all_keys)