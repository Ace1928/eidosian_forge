import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
def _run_inliner(self, state, inline_type, sig, template, arg_typs, expr, i, impl, block, work_list, is_method, inline_worker):
    do_inline = True
    if not inline_type.is_always_inline:
        from numba.core.typing.templates import _inline_info
        caller_inline_info = _inline_info(state.func_ir, state.typemap, state.calltypes, sig)
        iinfo = template._inline_overloads[arg_typs]['iinfo']
        if inline_type.has_cost_model:
            do_inline = inline_type.value(expr, caller_inline_info, iinfo)
        else:
            assert 'unreachable'
    if do_inline:
        if is_method:
            if not self._add_method_self_arg(state, expr):
                return False
        arg_typs = template._inline_overloads[arg_typs]['folded_args']
        iinfo = template._inline_overloads[arg_typs]['iinfo']
        freevars = iinfo.func_ir.func_id.func.__code__.co_freevars
        _, _, _, new_blocks = inline_worker.inline_ir(state.func_ir, block, i, iinfo.func_ir, freevars, arg_typs=arg_typs)
        if work_list is not None:
            for blk in new_blocks:
                work_list.append(blk)
        return True
    else:
        return False