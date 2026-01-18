import os
import sys
import textwrap
import types
import re
import warnings
import functools
import platform
from .._utils import set_module
from numpy.core.numerictypes import issubclass_, issubsctype, issubdtype
from numpy.core import ndarray, ufunc, asarray
import numpy as np
class _Deprecate:
    """
    Decorator class to deprecate old functions.

    Refer to `deprecate` for details.

    See Also
    --------
    deprecate

    """

    def __init__(self, old_name=None, new_name=None, message=None):
        self.old_name = old_name
        self.new_name = new_name
        self.message = message

    def __call__(self, func, *args, **kwargs):
        """
        Decorator call.  Refer to ``decorate``.

        """
        old_name = self.old_name
        new_name = self.new_name
        message = self.message
        if old_name is None:
            old_name = func.__name__
        if new_name is None:
            depdoc = '`%s` is deprecated!' % old_name
        else:
            depdoc = '`%s` is deprecated, use `%s` instead!' % (old_name, new_name)
        if message is not None:
            depdoc += '\n' + message

        @functools.wraps(func)
        def newfunc(*args, **kwds):
            warnings.warn(depdoc, DeprecationWarning, stacklevel=2)
            return func(*args, **kwds)
        newfunc.__name__ = old_name
        doc = func.__doc__
        if doc is None:
            doc = depdoc
        else:
            lines = doc.expandtabs().split('\n')
            indent = _get_indent(lines[1:])
            if lines[0].lstrip():
                doc = indent * ' ' + doc
            else:
                skip = len(lines[0]) + 1
                for line in lines[1:]:
                    if len(line) > indent:
                        break
                    skip += len(line) + 1
                doc = doc[skip:]
            depdoc = textwrap.indent(depdoc, ' ' * indent)
            doc = '\n\n'.join([depdoc, doc])
        newfunc.__doc__ = doc
        return newfunc