from __future__ import annotations
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Sequence, TypeVar
from .._C.libtriton.triton import ir
from . import semantic
def _add_atomic_docstr(name: str, has_cmp: bool=False) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = f'\n    Performs an atomic {name} at the memory location specified by :code:`pointer`.\n\n    Return the data stored at :code:`pointer` before the atomic operation.\n\n    :param pointer: The memory locations to operate on\n    :type pointer: Block of dtype=triton.PointerDType'
        if has_cmp:
            docstr += '\n    :param cmp: The values expected to be found in the atomic object\n    :type cmp: Block of dtype=pointer.dtype.element_ty'
        docstr += '\n    :param val: The values with which to perform the atomic operation\n    :type val: Block of dtype=pointer.dtype.element_ty\n    :param sem: Memory semantics to use ("ACQUIRE_RELEASE" (default),\n        "ACQUIRE", "RELEASE", or "RELAXED")\n    :type sem: str\n    :param scope: Scope of threads that observe synchronizing effect of the\n        atomic operation ("GPU" (default), "CTA", or "SYSTEM")\n    :type scope: str\n    '
        func.__doc__ = docstr
        return func
    return _decorator