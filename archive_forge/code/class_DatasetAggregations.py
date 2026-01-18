from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core import duck_array_ops
from xarray.core.options import OPTIONS
from xarray.core.types import Dims, Self
from xarray.core.utils import contains_only_chunked_or_numpy, module_available
class DatasetAggregations:
    __slots__ = ()

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, **kwargs: Any) -> Self:
        raise NotImplementedError()

    def count(self, dim: Dims=None, *, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        pandas.DataFrame.count
        dask.dataframe.DataFrame.count
        DataArray.count
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.count()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       int64 8B 5
        """
        return self.reduce(duck_array_ops.count, dim=dim, numeric_only=False, keep_attrs=keep_attrs, **kwargs)

    def all(self, dim: Dims=None, *, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        DataArray.all
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 78B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) bool 6B True True True True True False

        >>> ds.all()
        <xarray.Dataset> Size: 1B
        Dimensions:  ()
        Data variables:
            da       bool 1B False
        """
        return self.reduce(duck_array_ops.array_all, dim=dim, numeric_only=False, keep_attrs=keep_attrs, **kwargs)

    def any(self, dim: Dims=None, *, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        DataArray.any
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 78B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) bool 6B True True True True True False

        >>> ds.any()
        <xarray.Dataset> Size: 1B
        Dimensions:  ()
        Data variables:
            da       bool 1B True
        """
        return self.reduce(duck_array_ops.array_any, dim=dim, numeric_only=False, keep_attrs=keep_attrs, **kwargs)

    def max(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        DataArray.max
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.max()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 3.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.max(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan
        """
        return self.reduce(duck_array_ops.max, dim=dim, skipna=skipna, numeric_only=False, keep_attrs=keep_attrs, **kwargs)

    def min(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        DataArray.min
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.min()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 0.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.min(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan
        """
        return self.reduce(duck_array_ops.min, dim=dim, skipna=skipna, numeric_only=False, keep_attrs=keep_attrs, **kwargs)

    def mean(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        DataArray.mean
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.mean()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 1.6

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.mean(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan
        """
        return self.reduce(duck_array_ops.mean, dim=dim, skipna=skipna, numeric_only=True, keep_attrs=keep_attrs, **kwargs)

    def prod(self, dim: Dims=None, *, skipna: bool | None=None, min_count: int | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int or None, optional
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``prod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        DataArray.prod
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.prod()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 0.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.prod(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.prod(skipna=True, min_count=2)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 0.0
        """
        return self.reduce(duck_array_ops.prod, dim=dim, skipna=skipna, min_count=min_count, numeric_only=True, keep_attrs=keep_attrs, **kwargs)

    def sum(self, dim: Dims=None, *, skipna: bool | None=None, min_count: int | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int or None, optional
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``sum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        DataArray.sum
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.sum()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 8.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.sum(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.sum(skipna=True, min_count=2)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 8.0
        """
        return self.reduce(duck_array_ops.sum, dim=dim, skipna=skipna, min_count=min_count, numeric_only=True, keep_attrs=keep_attrs, **kwargs)

    def std(self, dim: Dims=None, *, skipna: bool | None=None, ddof: int=0, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        DataArray.std
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.std()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 1.02

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.std(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.std(skipna=True, ddof=1)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 1.14
        """
        return self.reduce(duck_array_ops.std, dim=dim, skipna=skipna, ddof=ddof, numeric_only=True, keep_attrs=keep_attrs, **kwargs)

    def var(self, dim: Dims=None, *, skipna: bool | None=None, ddof: int=0, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        DataArray.var
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.var()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 1.04

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.var(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.var(skipna=True, ddof=1)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 1.3
        """
        return self.reduce(duck_array_ops.var, dim=dim, skipna=skipna, ddof=ddof, numeric_only=True, keep_attrs=keep_attrs, **kwargs)

    def median(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        DataArray.median
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.median()
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.median(skipna=False)
        <xarray.Dataset> Size: 8B
        Dimensions:  ()
        Data variables:
            da       float64 8B nan
        """
        return self.reduce(duck_array_ops.median, dim=dim, skipna=skipna, numeric_only=True, keep_attrs=keep_attrs, **kwargs)

    def cumsum(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``cumsum`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``cumsum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``cumsum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``cumsum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.cumsum
        dask.array.cumsum
        DataArray.cumsum
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.cumsum()
        <xarray.Dataset> Size: 48B
        Dimensions:  (time: 6)
        Dimensions without coordinates: time
        Data variables:
            da       (time) float64 48B 1.0 3.0 6.0 6.0 8.0 8.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.cumsum(skipna=False)
        <xarray.Dataset> Size: 48B
        Dimensions:  (time: 6)
        Dimensions without coordinates: time
        Data variables:
            da       (time) float64 48B 1.0 3.0 6.0 6.0 8.0 nan
        """
        return self.reduce(duck_array_ops.cumsum, dim=dim, skipna=skipna, numeric_only=True, keep_attrs=keep_attrs, **kwargs)

    def cumprod(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> Self:
        """
        Reduce this Dataset's data by applying ``cumprod`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``cumprod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``cumprod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``cumprod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.cumprod
        dask.array.cumprod
        DataArray.cumprod
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 0, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("2001-01-01", freq="ME", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset> Size: 120B
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 48B 1.0 2.0 3.0 0.0 2.0 nan

        >>> ds.cumprod()
        <xarray.Dataset> Size: 48B
        Dimensions:  (time: 6)
        Dimensions without coordinates: time
        Data variables:
            da       (time) float64 48B 1.0 2.0 6.0 0.0 0.0 0.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.cumprod(skipna=False)
        <xarray.Dataset> Size: 48B
        Dimensions:  (time: 6)
        Dimensions without coordinates: time
        Data variables:
            da       (time) float64 48B 1.0 2.0 6.0 0.0 0.0 nan
        """
        return self.reduce(duck_array_ops.cumprod, dim=dim, skipna=skipna, numeric_only=True, keep_attrs=keep_attrs, **kwargs)