from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
class ChunkTensorBase(torch.Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, _elem, *_):
        return super().__new__(cls, _elem)

    @classmethod
    def __torch_dispatch__(cls, func, _, args=(), kwargs=None):

        def unwrap(t):
            return t._elem if isinstance(t, cls) else t
        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))