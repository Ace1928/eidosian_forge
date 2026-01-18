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
class ModuleContextCheckpointState:
    nn_modules: Dict[str, torch.nn.Module] = {}

    def __init__(self, nn_modules):
        self.nn_modules = nn_modules
    '\n    Produces a delta against another ModuleContextCheckpointState.\n\n    Returns None if no delta is found, otherwise, return a set() of mismatched\n    module key names.\n    '

    def diff(self, other):
        r = set(self.nn_modules.keys()).difference(set(other.nn_modules.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        return self.diff(other) is None