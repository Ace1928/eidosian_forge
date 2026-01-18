from __future__ import absolute_import, division, print_function
import itertools
import functools
import re
import types
from funcsigs.version import __version__
def _get_user_defined_method(cls, method_name, *nested):
    try:
        if cls is type:
            return
        meth = getattr(cls, method_name)
        for name in nested:
            meth = getattr(meth, name, meth)
    except AttributeError:
        return
    else:
        if not isinstance(meth, _NonUserDefinedCallables):
            return meth