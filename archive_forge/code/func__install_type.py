import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def _install_type(self, typingctx=None):
    """Constructs and installs a typing class for a DUFunc object in the
        input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
    if typingctx is None:
        typingctx = self._dispatcher.targetdescr.typing_context
    _ty_cls = type('DUFuncTyping_' + self.ufunc.__name__, (AbstractTemplate,), dict(key=self, generic=self._type_me))
    typingctx.insert_user_function(self, _ty_cls)
    self._install_ufunc_attributes(_ty_cls)
    self._install_ufunc_methods(_ty_cls)