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
class GlobalContextCheckpointState:
    global_state: Dict[str, Tuple[Callable, ...]] = {}

    def __init__(self, global_states):
        self.global_state = global_states
    '\n    Produces a delta against another GlobalContextCheckpointState.\n\n    Returns None if no delta is found, otherwise, return a set() of mismatched\n    global key names.\n    '

    def diff(self, other):
        r = set(self.global_state.keys()).difference(set(other.global_state.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        return self.diff(other) is None