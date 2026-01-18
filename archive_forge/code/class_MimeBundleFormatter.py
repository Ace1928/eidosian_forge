import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
class MimeBundleFormatter(BaseFormatter):
    """A Formatter for arbitrary mime-types.

    Unlike other `_repr_<mimetype>_` methods,
    `_repr_mimebundle_` should return mime-bundle data,
    either the mime-keyed `data` dictionary or the tuple `(data, metadata)`.
    Any mime-type is valid.

    To define the callables that compute the mime-bundle representation of your
    objects, define a :meth:`_repr_mimebundle_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    .. versionadded:: 6.1
    """
    print_method = ObjectName('_repr_mimebundle_')
    _return_type = dict

    def _check_return(self, r, obj):
        r = super(MimeBundleFormatter, self)._check_return(r, obj)
        if r is None:
            return ({}, {})
        if not isinstance(r, tuple):
            return (r, {})
        return r

    @catch_format_error
    def __call__(self, obj, include=None, exclude=None):
        """Compute the format for an object.

        Identical to parent's method but we pass extra parameters to the method.

        Unlike other _repr_*_ `_repr_mimebundle_` should allow extra kwargs, in
        particular `include` and `exclude`.
        """
        if self.enabled:
            try:
                printer = self.lookup(obj)
            except KeyError:
                pass
            else:
                return printer(obj)
            method = get_real_method(obj, self.print_method)
            if method is not None:
                return method(include=include, exclude=exclude)
            return None
        else:
            return None