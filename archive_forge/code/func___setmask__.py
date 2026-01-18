import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def __setmask__(self, mask, copy=False):
    """
        Set the mask.

        """
    idtype = self.dtype
    current_mask = self._mask
    if mask is masked:
        mask = True
    if current_mask is nomask:
        if mask is nomask:
            return
        current_mask = self._mask = make_mask_none(self.shape, idtype)
    if idtype.names is None:
        if self._hardmask:
            current_mask |= mask
        elif isinstance(mask, (int, float, np.bool_, np.number)):
            current_mask[...] = mask
        else:
            current_mask.flat = mask
    else:
        mdtype = current_mask.dtype
        mask = np.array(mask, copy=False)
        if not mask.ndim:
            if mask.dtype.kind == 'b':
                mask = np.array(tuple([mask.item()] * len(mdtype)), dtype=mdtype)
            else:
                mask = mask.astype(mdtype)
        else:
            try:
                mask = np.array(mask, copy=copy, dtype=mdtype)
            except TypeError:
                mask = np.array([tuple([m] * len(mdtype)) for m in mask], dtype=mdtype)
        if self._hardmask:
            for n in idtype.names:
                current_mask[n] |= mask[n]
        elif isinstance(mask, (int, float, np.bool_, np.number)):
            current_mask[...] = mask
        else:
            current_mask.flat = mask
    if current_mask.shape:
        current_mask.shape = self.shape
    return