from __future__ import annotations
from typing import (
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
def _create_delegator_method(name: str):

    def f(self, *args, **kwargs):
        return self._delegate_method(name, *args, **kwargs)
    f.__name__ = name
    f.__doc__ = getattr(delegate, accessor_mapping(name)).__doc__
    return f