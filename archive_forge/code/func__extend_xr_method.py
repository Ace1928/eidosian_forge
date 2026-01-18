import datetime
import functools
import importlib
import re
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np
import tree
import xarray as xr
from .. import __version__, utils
from ..rcparams import rcParams
def _extend_xr_method(func, doc=None, description='', examples='', see_also=''):
    """Make wrapper to extend methods from xr.Dataset to InferenceData Class.

    Parameters
    ----------
    func : callable
        An xr.Dataset function
    doc : str
        docstring for the func
    description : str
        the description of the func to be added in docstring
    examples : str
        the examples of the func to be added in docstring
    see_also : str, list
        the similar methods of func to be included in See Also section of docstring

    """

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        _filter = kwargs.pop('filter_groups', None)
        _groups = kwargs.pop('groups', None)
        _inplace = kwargs.pop('inplace', False)
        out = self if _inplace else deepcopy(self)
        groups = self._group_names(_groups, _filter)
        for group in groups:
            xr_data = getattr(out, group)
            xr_data = func(xr_data, *args, **kwargs)
            setattr(out, group, xr_data)
        return None if _inplace else out
    description_default = '{method_name} method is extended from xarray.Dataset methods.\n\n    {description}\n\n    For more info see :meth:`xarray:xarray.Dataset.{method_name}`.\n    In addition to the arguments available in the original method, the following\n    ones are added by ArviZ to adapt the method to being called on an ``InferenceData`` object.\n    '.format(description=description, method_name=func.__name__)
    params = '\n    Other Parameters\n    ----------------\n    groups: str or list of str, optional\n        Groups where the selection is to be applied. Can either be group names\n        or metagroup names.\n    filter_groups: {None, "like", "regex"}, optional, default=None\n        If `None` (default), interpret groups as the real group or metagroup names.\n        If "like", interpret groups as substrings of the real group or metagroup names.\n        If "regex", interpret groups as regular expressions on the real group or\n        metagroup names. A la `pandas.filter`.\n    inplace: bool, optional\n        If ``True``, modify the InferenceData object inplace,\n        otherwise, return the modified copy.\n    '
    if not isinstance(see_also, str):
        see_also = '\n'.join(see_also)
    see_also_basic = '\n    See Also\n    --------\n    xarray.Dataset.{method_name}\n    {custom_see_also}\n    '.format(method_name=func.__name__, custom_see_also=see_also)
    wrapped.__doc__ = description_default + params + examples + see_also_basic if doc is None else doc
    return wrapped