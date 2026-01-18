import collections
import dataclasses
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.fx
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor, istype, iter_contains
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable
@staticmethod
@functools.lru_cache(None)
def _patch_once():
    try:
        from transformers.file_utils import ModelOutput
        for obj in ModelOutput.__dict__.values():
            if callable(obj):
                skip_code(obj.__code__)
    except ImportError:
        pass
    try:
        from diffusers.utils import BaseOutput
        for obj in BaseOutput.__dict__.values():
            if callable(obj):
                skip_code(obj.__code__)
    except ImportError:
        pass