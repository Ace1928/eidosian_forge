import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def _hoist_internal(inst, dep_on_param, call_table, hoisted, not_hoisted, typemap, stored_arrays):
    if inst.target.name in stored_arrays:
        not_hoisted.append((inst, 'stored array'))
        if config.DEBUG_ARRAY_OPT >= 1:
            print('Instruction', inst, ' could not be hoisted because the created array is stored.')
        return False
    uses = set()
    visit_vars_inner(inst.value, find_vars, uses)
    diff = uses.difference(dep_on_param)
    if config.DEBUG_ARRAY_OPT >= 1:
        print('_hoist_internal:', inst, 'uses:', uses, 'diff:', diff)
    if len(diff) == 0 and is_pure(inst.value, None, call_table):
        if config.DEBUG_ARRAY_OPT >= 1:
            print('Will hoist instruction', inst, typemap[inst.target.name])
        hoisted.append(inst)
        if not isinstance(typemap[inst.target.name], types.npytypes.Array):
            dep_on_param += [inst.target.name]
        return True
    elif len(diff) > 0:
        not_hoisted.append((inst, 'dependency'))
        if config.DEBUG_ARRAY_OPT >= 1:
            print('Instruction', inst, ' could not be hoisted because of a dependency.')
    else:
        not_hoisted.append((inst, 'not pure'))
        if config.DEBUG_ARRAY_OPT >= 1:
            print('Instruction', inst, " could not be hoisted because it isn't pure.")
    return False