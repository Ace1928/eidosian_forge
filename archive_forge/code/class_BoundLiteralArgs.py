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
class BoundLiteralArgs(collections.namedtuple('BoundLiteralArgs', ['pysig', 'literal_args'])):
    """
    This class is usually created by SentryLiteralArgs.
    """

    def bind(self, *args, **kwargs):
        """Bind to argument types.
        """
        return sentry_literal_args(self.pysig, self.literal_args, args, kwargs)