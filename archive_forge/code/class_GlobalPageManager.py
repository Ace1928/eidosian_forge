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
class GlobalPageManager:
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def prefetch_all(self, to_cpu=False):
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)