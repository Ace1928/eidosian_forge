import functools
from collections import defaultdict
from typing import Callable, Dict
import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload
class PhiloxState:
    """
    Represents a PhiloxRngState - (seed, offset) where offset = base_offset +
    relative_offset. seed and base_offset basically point to the rng state just
    before tracing starts. relative offset tracks the totally consumed offset at
    trace time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.seed = torch.tensor(())
        self.base_offset = torch.tensor(())
        self.relative_offset = 0
        self.offset_advanced_alteast_once = False

    def validate_state(self):
        assert self.seed.numel() != 0 and self.base_offset.numel() != 0

    def advance_offset(self, consumed_offset):
        self.offset_advanced_alteast_once = True
        self.relative_offset = self.relative_offset + consumed_offset

    def set_state(self, seed, base_offset, relative_offset=0):
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset

    def get_state_as_tuple(self):
        self.validate_state()
        return (self.seed, self.base_offset + self.relative_offset)

    def get_state_as_tensor(self):
        self.validate_state()
        return torch.stack([self.seed, self.base_offset + self.relative_offset])

    def set_state_from_tensor(self, state):
        self.seed, self.base_offset = torch.unbind(state)
        self.relative_offset = 0