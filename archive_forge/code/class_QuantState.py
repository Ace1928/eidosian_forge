import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
class QuantState:
    """container for quantization state components to work with Params4bit and similar classes"""
    valid_quant_types = ('fp4', 'nf4')
    valid_qs_type_keys = [f'bitsandbytes__{x}' for x in valid_quant_types]
    valid_qs_keys = ['absmax', 'quant_map', 'nested_absmax', 'nested_quant_map', 'quant_state', 'quant_type', 'blocksize', 'dtype', 'shape', 'nested_blocksize', 'nested_dtype', 'nested_offset']

    def __init__(self, absmax, shape=None, code=None, blocksize=None, quant_type=None, dtype=None, offset=None, state2=None):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __get_item__(self, idx):
        """
        ensures compatibility with older quant state scheme with nested lists.
        assumes the following layout:
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
        state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
        """
        if self.nested:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, [self.offset, self.state2], self.quant_type]
        else:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
        return list_repr[idx]

    @classmethod
    def from_dict(cls, qs_dict: Dict[str, Any], device: torch.device) -> 'QuantState':
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, torch.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """
        qs_key = [k for k, v in qs_dict.items() if 'quant_state' in k and isinstance(v, torch.Tensor)]
        if not len(qs_key) and 'quant_type' not in qs_dict:
            raise ValueError('Expected packed or unpacked quant_state items, found neither')
        elif len(qs_key) != 1 or qs_key[0].split('.')[-1] not in cls.valid_qs_type_keys:
            raise ValueError(f'There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.')
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))
        qs_dict = {k.split('.')[-1]: v for k, v in qs_dict.items()}
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)
        if 'nested_absmax' in qs_dict:
            offset = torch.tensor(float(qs_dict['nested_offset'])).to(device)
            state2 = cls(absmax=qs_dict['nested_absmax'].to(device), blocksize=qs_dict['nested_blocksize'], code=qs_dict['nested_quant_map'].to(device), dtype=getattr(torch, qs_dict['nested_dtype']))
        else:
            offset, state2 = (None, None)
        quant_state = cls(quant_type=qs_dict['quant_type'], absmax=qs_dict['absmax'].to(device), blocksize=qs_dict['blocksize'], code=qs_dict['quant_map'].to(device), dtype=getattr(torch, qs_dict['dtype']), shape=torch.Size(qs_dict['shape']) if qs_dict['shape'] is not None else None, offset=offset, state2=state2)
        return quant_state

    def as_dict(self, packed=False):
        """
        returns dict of tensors and strings to use in serialization via _save_to_state_dict()
        param: packed -- returns dict[str, torch.Tensor] for state_dict fit for safetensors saving
        """
        qs_dict = {'quant_type': self.quant_type, 'absmax': self.absmax, 'blocksize': self.blocksize, 'quant_map': self.code, 'dtype': str(self.dtype).strip('torch.'), 'shape': tuple(self.shape)}
        if self.nested:
            qs_dict.update({'nested_absmax': self.state2.absmax, 'nested_blocksize': self.state2.blocksize, 'nested_quant_map': self.state2.code.clone(), 'nested_dtype': str(self.state2.dtype).strip('torch.'), 'nested_offset': self.offset.item()})
        if not packed:
            return qs_dict
        qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
        non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
        qs_packed_dict['quant_state.' + 'bitsandbytes__' + self.quant_type] = pack_dict_to_tensor(non_tensor_dict)
        return qs_packed_dict

    def to(self, device):
        self.absmax = self.absmax.to(device)
        if self.nested:
            self.offset = self.offset.to(device)
            self.state2.absmax = self.state2.absmax.to(device)
            self.state2.code = self.state2.code.to(device)

    def __eq__(self, other):
        if not isinstance(other, QuantState):
            return False
        return torch.allclose(self.absmax, other.absmax, atol=1e-06) and self.shape == other.shape and torch.allclose(self.code, other.code, atol=1e-06) and (self.dtype == other.dtype) and (self.blocksize == other.blocksize) and (self.quant_type == other.quant_type) and (self.offset == other.offset if self.offset is not None and other.offset is not None else self.offset is other.offset) and (self.state2 == other.state2 if self.state2 is not None and other.state2 is not None else self.state2 is other.state2)