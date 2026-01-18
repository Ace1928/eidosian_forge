import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
class Parfor(ir.Expr, ir.Stmt):
    id_counter = 0

    def __init__(self, loop_nests, init_block, loop_body, loc, index_var, equiv_set, pattern, flags, *, no_sequential_lowering=False, races=set()):
        super(Parfor, self).__init__(op='parfor', loc=loc)
        self.id = type(self).id_counter
        type(self).id_counter += 1
        self.loop_nests = loop_nests
        self.init_block = init_block
        self.loop_body = loop_body
        self.index_var = index_var
        self.params = None
        self.equiv_set = equiv_set
        assert len(pattern) > 1
        self.patterns = [pattern]
        self.flags = flags
        self.no_sequential_lowering = no_sequential_lowering
        self.races = races
        self.redvars = []
        self.reddict = {}
        self.lowerer = None
        if config.DEBUG_ARRAY_OPT_STATS:
            fmt = "Parallel for-loop #{} is produced from pattern '{}' at {}"
            print(fmt.format(self.id, pattern, loc))

    def __repr__(self):
        return 'id=' + str(self.id) + repr(self.loop_nests) + repr(self.loop_body) + repr(self.index_var)

    def get_loop_nest_vars(self):
        return [x.index_variable for x in self.loop_nests]

    def list_vars(self):
        """list variables used (read/written) in this parfor by
        traversing the body and combining block uses.
        """
        all_uses = []
        for l, b in self.loop_body.items():
            for stmt in b.body:
                all_uses += stmt.list_vars()
        for loop in self.loop_nests:
            all_uses += loop.list_vars()
        for stmt in self.init_block.body:
            all_uses += stmt.list_vars()
        return all_uses

    def get_shape_classes(self, var, typemap=None):
        """get the shape classes for a given variable.
        If a typemap is specified then use it for type resolution
        """
        if typemap is not None:
            save_typemap = self.equiv_set.typemap
            self.equiv_set.typemap = typemap
        res = self.equiv_set.get_shape_classes(var)
        if typemap is not None:
            self.equiv_set.typemap = save_typemap
        return res

    def dump(self, file=None):
        file = file or sys.stdout
        print('begin parfor {}'.format(self.id).center(20, '-'), file=file)
        print('index_var = ', self.index_var, file=file)
        print('params = ', self.params, file=file)
        print('races = ', self.races, file=file)
        for loopnest in self.loop_nests:
            print(loopnest, file=file)
        print('init block:', file=file)
        self.init_block.dump(file)
        for offset, block in sorted(self.loop_body.items()):
            print('label %s:' % (offset,), file=file)
            block.dump(file)
        print('end parfor {}'.format(self.id).center(20, '-'), file=file)

    def validate_params(self, typemap):
        """
        Check that Parfors params are of valid types.
        """
        if self.params is None:
            msg = 'Cannot run parameter validation on a Parfor with params not set'
            raise ValueError(msg)
        for p in self.params:
            ty = typemap.get(p)
            if ty is None:
                msg = 'Cannot validate parameter %s, there is no type information available'
                raise ValueError(msg)
            if isinstance(ty, types.BaseTuple):
                if ty.count > config.PARFOR_MAX_TUPLE_SIZE:
                    msg = 'Use of a tuple (%s) of length %d in a parallel region exceeds the maximum supported tuple size.  Since Generalized Universal Functions back parallel regions and those do not support tuples, tuples passed to parallel regions are unpacked if their size is below a certain threshold, currently configured to be %d. This threshold can be modified using the Numba environment variable NUMBA_PARFOR_MAX_TUPLE_SIZE.'
                    raise errors.UnsupportedParforsError(msg % (p, ty.count, config.PARFOR_MAX_TUPLE_SIZE), self.loc)