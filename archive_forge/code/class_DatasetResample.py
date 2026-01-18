from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core._aggregations import (
from xarray.core.groupby import DataArrayGroupByBase, DatasetGroupByBase, GroupBy
from xarray.core.types import Dims, InterpOptions, T_Xarray
class DatasetResample(Resample['Dataset'], DatasetGroupByBase, DatasetResampleAggregations):
    """DatasetGroupBy object specialized to resampling a specified dimension"""

    def map(self, func: Callable[..., Any], args: tuple[Any, ...]=(), shortcut: bool | None=None, **kwargs: Any) -> Dataset:
        """Apply a function over each Dataset in the groups generated for
        resampling and concatenate them together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each sub-dataset.
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset
            The result of splitting, applying and combining this dataset.
        """
        return self._map_maybe_warn(func, args, shortcut, warn_squeeze=True, **kwargs)

    def _map_maybe_warn(self, func: Callable[..., Any], args: tuple[Any, ...]=(), shortcut: bool | None=None, warn_squeeze: bool=True, **kwargs: Any) -> Dataset:
        applied = (func(ds, *args, **kwargs) for ds in self._iter_grouped(warn_squeeze))
        combined = self._combine(applied)
        if self._dim in combined.coords:
            combined = combined.drop_vars(self._dim)
        if RESAMPLE_DIM in combined.dims:
            combined = combined.rename({RESAMPLE_DIM: self._dim})
        return combined

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataSetResample.map
        """
        warnings.warn('Resample.apply may be deprecated in the future. Using Resample.map is encouraged', PendingDeprecationWarning, stacklevel=2)
        return self.map(func=func, shortcut=shortcut, args=args, **kwargs)

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> Dataset:
        """Reduce the items in this group by applying `func` along the
        pre-defined resampling dimension.

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        return super().reduce(func=func, dim=dim, axis=axis, keep_attrs=keep_attrs, keepdims=keepdims, shortcut=shortcut, **kwargs)

    def _reduce_without_squeeze_warn(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> Dataset:
        return super()._reduce_without_squeeze_warn(func=func, dim=dim, axis=axis, keep_attrs=keep_attrs, keepdims=keepdims, shortcut=shortcut, **kwargs)

    def asfreq(self) -> Dataset:
        """Return values of original object at the new up-sampling frequency;
        essentially a re-index with new times set to NaN.

        Returns
        -------
        resampled : Dataset
        """
        self._obj = self._drop_coords()
        return self.mean(None if self._dim is None else [self._dim])