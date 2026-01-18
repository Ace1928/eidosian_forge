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
class TraceId(NamedTuple):
    compile_id: CompileId
    attempt: int

    def __str__(self):
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f'{self.compile_id}_{self.attempt}'