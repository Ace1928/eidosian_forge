from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def add_colorbar(self, **kwargs: Any) -> None:
    """Draw a colorbar."""
    kwargs = kwargs.copy()
    if self._cmap_extend is not None:
        kwargs.setdefault('extend', self._cmap_extend)
    if hasattr(self._mappables[-1], 'extend'):
        kwargs.pop('extend', None)
    if 'label' not in kwargs:
        from xarray import DataArray
        assert isinstance(self.data, DataArray)
        kwargs.setdefault('label', label_from_attrs(self.data))
    self.cbar = self.fig.colorbar(self._mappables[-1], ax=list(self.axs.flat), **kwargs)