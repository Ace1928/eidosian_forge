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
class CompileContext:

    @staticmethod
    def get() -> CompileContext:
        assert _TLS.compile_context is not None
        return _TLS.compile_context

    @staticmethod
    def try_get() -> Optional[CompileContext]:
        return getattr(_TLS, 'compile_context', None)

    def __init__(self, compile_id):
        assert compile_id is None or isinstance(compile_id, CompileId)
        self.compile_id: Optional[CompileId] = compile_id
        self.attempt = 0

    @staticmethod
    def current_compile_id():
        self = CompileContext.try_get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id():
        self = CompileContext.try_get()
        if self is None:
            return None
        if self.compile_id is None:
            return None
        return TraceId(self.compile_id, self.attempt)