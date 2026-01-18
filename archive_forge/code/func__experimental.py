import inspect
import re
import types
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, Union
def _experimental(api: C, api_type: str) -> C:
    indent = _get_min_indent_of_docstring(api.__doc__)
    notice = indent + f'.. Note:: Experimental: This {api_type} may change or be removed in a future release without warning.\n\n'
    if api_type == 'property':
        api.__doc__ = api.__doc__ + '\n\n' + notice if api.__doc__ else notice
    else:
        api.__doc__ = notice + api.__doc__ if api.__doc__ else notice
    return api