from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
def _get_jit_decorator(self):
    """Gets a jit decorator suitable for the current target"""
    from numba.core.target_extension import target_registry, get_local_target, jit_registry
    jitter_str = self.metadata.get('target', 'generic')
    jitter = jit_registry.get(jitter_str, None)
    if jitter is None:
        target_class = target_registry.get(jitter_str, None)
        if target_class is None:
            msg = ("Unknown target '{}', has it been ", 'registered?')
            raise ValueError(msg.format(jitter_str))
        target_hw = get_local_target(self.context)
        if not issubclass(target_hw, target_class):
            msg = 'No overloads exist for the requested target: {}.'
        jitter = jit_registry[target_hw]
    if jitter is None:
        raise ValueError('Cannot find a suitable jit decorator')
    return jitter