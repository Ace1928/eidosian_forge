from __future__ import annotations
import contextlib
import dataclasses
import enum
import functools
import logging
import threading
import traceback
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
import torch
from torch.utils import _pytree as pytree
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakTensorKeyDictionary
@dataclasses.dataclass
class DuplicateInputs(GuardEnvExpr):
    input_source_a: Source
    input_source_b: Source

    def __post_init__(self):
        assert self.input_source_a != self.input_source_b