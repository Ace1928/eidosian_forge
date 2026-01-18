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
def _create_shape_signature(get_shape_classes, num_inputs, num_reductions, args, func_sig, races, typemap):
    """Create shape signature for GUFunc
    """
    if config.DEBUG_ARRAY_OPT:
        print('_create_shape_signature', num_inputs, num_reductions, args, races)
        for i in args[1:]:
            print('argument', i, type(i), get_shape_classes(i, typemap=typemap))
    num_inouts = len(args) - num_reductions
    classes = [get_shape_classes(var, typemap=typemap) if var not in races else (-1,) for var in args[1:]]
    class_set = set()
    for _class in classes:
        if _class:
            for i in _class:
                class_set.add(i)
    max_class = max(class_set) + 1 if class_set else 0
    classes.insert(0, (max_class,))
    class_set.add(max_class)
    thread_num_class = max_class + 1
    class_set.add(thread_num_class)
    class_map = {}
    alphabet = ord('a')
    for n in class_set:
        if n >= 0:
            class_map[n] = chr(alphabet)
            alphabet += 1
    threadcount_ordinal = chr(alphabet)
    alpha_dict = {'latest_alpha': alphabet}

    def bump_alpha(c, class_map):
        if c >= 0:
            return class_map[c]
        else:
            alpha_dict['latest_alpha'] += 1
            return chr(alpha_dict['latest_alpha'])
    gu_sin = []
    gu_sout = []
    count = 0
    syms_sin = ()
    if config.DEBUG_ARRAY_OPT:
        print('args', args)
        print('classes', classes)
        print('threadcount_ordinal', threadcount_ordinal)
    for cls, arg in zip(classes, args):
        count = count + 1
        if cls:
            dim_syms = tuple((bump_alpha(c, class_map) for c in cls))
        else:
            dim_syms = ()
        if count > num_inouts:
            gu_sin.append(tuple([threadcount_ordinal] + list(dim_syms[1:])))
        else:
            gu_sin.append(dim_syms)
            syms_sin += dim_syms
    return (gu_sin, gu_sout)