import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
def _getsetitem_gen(getset):
    _dunder_meth = '__%s__' % getset
    op = getattr(operator, getset)

    @templates.infer_global(op)
    class GetSetItem(templates.AbstractTemplate):

        def generic(self, args, kws):
            instance = args[0]
            if isinstance(instance, types.ClassInstanceType) and _dunder_meth in instance.jit_methods:
                meth = instance.jit_methods[_dunder_meth]
                disp_type = types.Dispatcher(meth)
                sig = disp_type.get_call_type(self.context, args, kws)
                return sig
    imputils.lower_builtin((types.ClassInstanceType, _dunder_meth), types.ClassInstanceType, types.VarArg(types.Any))(get_imp())
    imputils.lower_builtin(op, types.ClassInstanceType, types.VarArg(types.Any))(get_imp())