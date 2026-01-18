from __future__ import annotations, division
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
def _call_hook(self, key, signature, device, constants, num_warps, num_ctas, num_stages, enable_warp_specialization, enable_fp_fusion, extern_libs, configs):
    if JITFunction.cache_hook is None:
        return False
    name = self.fn.__name__
    module = self.fn.__module__
    arg_reprs = ', '.join([f'{param.name}: {ty}' for param, ty in zip(self.params, key[1])])
    repr = f'{name}[num_warps={num_warps}, num_ctas={num_ctas}, num_stages={num_stages}, enable_warp_specialization={enable_warp_specialization}, enable_fp_fusion={enable_fp_fusion}]({arg_reprs})'
    key = str(key)

    class LegacyCompiler:

        def __init__(self, module, name):
            self.module = module
            self.name = name
            pass
    kwargs = dict(signature=signature, device=device, constants=constants, num_warps=num_warps, num_ctas=num_ctas, num_stages=num_stages, enable_warp_specialization=enable_warp_specialization, enable_fp_fusion=enable_fp_fusion, extern_libs=extern_libs, configs=configs)
    return JITFunction.cache_hook(key=key, repr=repr, fn=LegacyCompiler(module, name), compile={'key': key, **kwargs}, is_manual_warmup=False, already_compiled=False)