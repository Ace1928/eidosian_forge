from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
class CUDARngStateHelper:

    @staticmethod
    def get_torch_state_as_tuple(fake_mode=nullcontext()):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        with fake_mode:
            seed = torch.tensor(torch.cuda.initial_seed())
            offset = torch.tensor(torch.cuda._get_rng_state_offset())
            return (seed, offset)

    @staticmethod
    def set_torch_state_tensor(seed, offset):
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        new_state = torch.cat([seed_portion, offset_portion])
        torch.cuda.set_rng_state(new_state)

    @staticmethod
    def set_new_offset(relative_offset):
        torch.cuda._set_rng_state_offset(relative_offset.item())