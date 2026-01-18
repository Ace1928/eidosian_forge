import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from ..modules import Module
class SpectralNormStateDictHook:

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError(f"Unexpected key in metadata['spectral_norm']: {key}")
        local_metadata['spectral_norm'][key] = self.fn._version