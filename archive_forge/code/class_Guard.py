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
class Guard:
    originating_source: Source
    create_fn: Callable[[GuardBuilderBase, Guard], None]
    guard_types: Optional[List[str]] = None
    code_list: Optional[List[str]] = None
    obj_weakref: Optional[object] = None
    guarded_class_weakref: Optional[type] = None
    stack = None
    user_stack = None
    _hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.name, self.source, id(self.create_fn)))
        return self._hash

    def sort_key(self):
        return (self.source.value if self.source else -1, len(self.name), self.name, self.inner_create_fn().__code__.co_firstlineno)

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    def inner_create_fn(self):
        if isinstance(self.create_fn, functools.partial):
            return self.create_fn.func
        else:
            return self.create_fn

    @property
    def name(self) -> str:
        return self.originating_source.name()

    @property
    def source(self) -> GuardSource:
        return self.originating_source.guard_source()

    @staticmethod
    def weakref_to_str(obj_weakref):
        """
        This is a workaround of a Python weakref bug.

        `obj_weakref` is instance returned by `weakref.ref`,
        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:

            class MyConfig(dict):
                def __getattr__(self, x):
                    return self[x]

            obj = MyConfig(offset=5)
            obj_weakref = weakref.ref(obj)
            str(obj_weakref)  # raise error: KeyError: '__name__'
        """
        if isinstance(obj_weakref, weakref.ReferenceType):
            obj = obj_weakref()
            if obj is not None:
                return f"<weakref at {hex(id(obj_weakref))}; to '{obj.__class__.__name__}' at {hex(id(obj))}>"
            else:
                return f'<weakref at {hex(id(obj_weakref))}; dead>'
        else:
            return str(obj_weakref)

    def __repr__(self):
        s = f"\n        {(self.source.name.lower() if self.source else '')} {repr(self.name)} {self.inner_create_fn().__name__}\n        {{\n            'guard_types': {self.guard_types},\n            'code': {self.code_list},\n            'obj_weakref': {self.weakref_to_str(self.obj_weakref)}\n            'guarded_class': {self.guarded_class_weakref}\n        }}\n        "
        return s

    def __str__(self):
        output = f'Name: {repr(self.name)}\n'
        source = self.source.name.lower() if self.source else ''
        output += f'    Source: {source}\n'
        output += f'    Create Function: {self.inner_create_fn().__name__}\n'
        output += f'    Guard Types: {self.guard_types}\n'
        output += f'    Code List: {self.code_list}\n'
        output += f'    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n'
        output += f'    Guarded Class Weakref: {self.guarded_class_weakref}\n'
        return output

    def create(self, builder: GuardBuilderBase):
        try:
            return self.create_fn(builder, self)
        except Exception:
            log.error('Error while creating guard:\n%s', str(self).rstrip())
            if self.stack:
                log.error('Created at:\n%s', ''.join(self.stack.format()[-4:]).rstrip())
            raise

    def is_nn_module(self):
        return self.source.is_nn_module()

    def is_fsdp_module(self):
        return self.source.is_fsdp_module()

    def is_local(self):
        return self.source.is_local()

    def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref):
        if not self.guard_types:
            self.guard_types = list()
        self.guard_types.append(guard_type)
        assert self.guarded_class_weakref in (guarded_class, None), 'Guarded class id must be identical, or None'
        self.guarded_class_weakref = guarded_class
        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)
        assert self.obj_weakref in (obj_weakref, None), 'Guarded object must be identical, or None'
        self.obj_weakref = obj_weakref