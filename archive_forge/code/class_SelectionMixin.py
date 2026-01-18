from __future__ import annotations
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import (
class SelectionMixin(Generic[NDFrameT]):
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """
    obj: NDFrameT
    _selection: IndexLabel | None = None
    exclusions: frozenset[Hashable]
    _internal_names = ['_cache', '__setstate__']
    _internal_names_set = set(_internal_names)

    @final
    @property
    def _selection_list(self):
        if not isinstance(self._selection, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            return [self._selection]
        return self._selection

    @cache_readonly
    def _selected_obj(self):
        if self._selection is None or isinstance(self.obj, ABCSeries):
            return self.obj
        else:
            return self.obj[self._selection]

    @final
    @cache_readonly
    def ndim(self) -> int:
        return self._selected_obj.ndim

    @final
    @cache_readonly
    def _obj_with_exclusions(self):
        if isinstance(self.obj, ABCSeries):
            return self.obj
        if self._selection is not None:
            return self.obj._getitem_nocopy(self._selection_list)
        if len(self.exclusions) > 0:
            return self.obj._drop_axis(self.exclusions, axis=1, only_slice=True)
        else:
            return self.obj

    def __getitem__(self, key):
        if self._selection is not None:
            raise IndexError(f'Column(s) {self._selection} already selected')
        if isinstance(key, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            if len(self.obj.columns.intersection(key)) != len(set(key)):
                bad_keys = list(set(key).difference(self.obj.columns))
                raise KeyError(f'Columns not found: {str(bad_keys)[1:-1]}')
            return self._gotitem(list(key), ndim=2)
        else:
            if key not in self.obj:
                raise KeyError(f'Column not found: {key}')
            ndim = self.obj[key].ndim
            return self._gotitem(key, ndim=ndim)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        raise AbstractMethodError(self)

    @final
    def _infer_selection(self, key, subset: Series | DataFrame):
        """
        Infer the `selection` to pass to our constructor in _gotitem.
        """
        selection = None
        if subset.ndim == 2 and (lib.is_scalar(key) and key in subset or lib.is_list_like(key)):
            selection = key
        elif subset.ndim == 1 and lib.is_scalar(key) and (key == subset.name):
            selection = key
        return selection

    def aggregate(self, func, *args, **kwargs):
        raise AbstractMethodError(self)
    agg = aggregate