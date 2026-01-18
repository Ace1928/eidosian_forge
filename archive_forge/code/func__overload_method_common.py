import os
import uuid
import weakref
import collections
import functools
import numba
from numba.core import types, errors, utils, config
from numba.core.typing.typeof import typeof_impl  # noqa: F401
from numba.core.typing.asnumbatype import as_numba_type  # noqa: F401
from numba.core.typing.templates import infer, infer_getattr  # noqa: F401
from numba.core.imputils import (  # noqa: F401
from numba.core.datamodel import models   # noqa: F401
from numba.core.datamodel import register_default as register_model  # noqa: F401, E501
from numba.core.pythonapi import box, unbox, reflect, NativeValue  # noqa: F401
from numba._helperlib import _import_cython_function  # noqa: F401
from numba.core.serialize import ReduceMixin
def _overload_method_common(typ, attr, **kwargs):
    """Common code for overload_method and overload_classmethod
    """
    from numba.core.typing.templates import make_overload_method_template

    def decorate(overload_func):
        copied_kwargs = kwargs.copy()
        template = make_overload_method_template(typ, attr, overload_func, inline=copied_kwargs.pop('inline', 'never'), prefer_literal=copied_kwargs.pop('prefer_literal', False), **copied_kwargs)
        infer_getattr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func
    return decorate