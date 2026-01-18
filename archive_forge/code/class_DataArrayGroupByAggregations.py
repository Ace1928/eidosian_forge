from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core import duck_array_ops
from xarray.core.options import OPTIONS
from xarray.core.types import Dims, Self
from xarray.core.utils import contains_only_chunked_or_numpy, module_available
class DataArrayGroupByAggregations:
    _obj: DataArray

    def _reduce_without_squeeze_warn(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> DataArray:
        raise NotImplementedError()

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, **kwargs: Any) -> DataArray:
        raise NotImplementedError()

    def _flox_reduce(self, dim: Dims, **kwargs: Any) -> DataArray:
        raise NotImplementedError()

    def count(self, dim: Dims=None, *, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        pandas.DataFrame.count
        dask.dataframe.DataFrame.count
        DataArray.count
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").count()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([1, 2, 2])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='count', dim=dim, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.count, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def all(self, dim: Dims=None, *, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        DataArray.all
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 6B
        array([ True,  True,  True,  True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").all()
        <xarray.DataArray (labels: 3)> Size: 3B
        array([False,  True,  True])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='all', dim=dim, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.array_all, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def any(self, dim: Dims=None, *, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        DataArray.any
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 6B
        array([ True,  True,  True,  True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").any()
        <xarray.DataArray (labels: 3)> Size: 3B
        array([ True,  True,  True])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='any', dim=dim, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.array_any, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def max(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        DataArray.max
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").max()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([1., 2., 3.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").max(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan,  2.,  3.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='max', dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.max, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

    def min(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        DataArray.min
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").min()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([1., 2., 0.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").min(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan,  2.,  0.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='min', dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.min, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

    def mean(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        DataArray.mean
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").mean()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([1. , 2. , 1.5])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").mean(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan, 2. , 1.5])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='mean', dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.mean, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

    def prod(self, dim: Dims=None, *, skipna: bool | None=None, min_count: int | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        DataArray.prod
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").prod()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([1., 4., 0.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").prod(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan,  4.,  0.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.groupby("labels").prod(skipna=True, min_count=2)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan,  4.,  0.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='prod', dim=dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.prod, dim=dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs, **kwargs)

    def sum(self, dim: Dims=None, *, skipna: bool | None=None, min_count: int | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        DataArray.sum
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").sum()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([1., 4., 3.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").sum(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan,  4.,  3.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.groupby("labels").sum(skipna=True, min_count=2)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan,  4.,  3.])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='sum', dim=dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.sum, dim=dim, skipna=skipna, min_count=min_count, keep_attrs=keep_attrs, **kwargs)

    def std(self, dim: Dims=None, *, skipna: bool | None=None, ddof: int=0, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        DataArray.std
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").std()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([0. , 0. , 1.5])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").std(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan, 0. , 1.5])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.groupby("labels").std(skipna=True, ddof=1)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([       nan, 0.        , 2.12132034])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='std', dim=dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.std, dim=dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs, **kwargs)

    def var(self, dim: Dims=None, *, skipna: bool | None=None, ddof: int=0, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        DataArray.var
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").var()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([0.  , 0.  , 2.25])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").var(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([ nan, 0.  , 2.25])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.groupby("labels").var(skipna=True, ddof=1)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan, 0. , 4.5])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        if flox_available and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj):
            return self._flox_reduce(func='var', dim=dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs, **kwargs)
        else:
            return self._reduce_without_squeeze_warn(duck_array_ops.var, dim=dim, skipna=skipna, ddof=ddof, keep_attrs=keep_attrs, **kwargs)

    def median(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        DataArray.median
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").median()
        <xarray.DataArray (labels: 3)> Size: 24B
        array([1. , 2. , 1.5])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").median(skipna=False)
        <xarray.DataArray (labels: 3)> Size: 24B
        array([nan, 2. , 1.5])
        Coordinates:
          * labels   (labels) object 24B 'a' 'b' 'c'
        """
        return self._reduce_without_squeeze_warn(duck_array_ops.median, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

    def cumsum(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``cumsum`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``cumsum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``cumsum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.cumsum
        dask.array.cumsum
        DataArray.cumsum
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").cumsum()
        <xarray.DataArray (time: 6)> Size: 48B
        array([1., 2., 3., 3., 4., 1.])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").cumsum(skipna=False)
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  3.,  4., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        """
        return self._reduce_without_squeeze_warn(duck_array_ops.cumsum, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

    def cumprod(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``cumprod`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``cumprod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the GroupBy dimensions.
            If "...", will reduce over all dimensions.
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
        reduced : DataArray
            New DataArray with ``cumprod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.cumprod
        dask.array.cumprod
        DataArray.cumprod
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up groupby computations,
        especially with dask arrays. Xarray will use flox by default if installed.
        Pass flox-specific keyword arguments in ``**kwargs``.
        See the `flox documentation <https://flox.readthedocs.io>`_ for more.

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
        >>> da
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").cumprod()
        <xarray.DataArray (time: 6)> Size: 48B
        array([1., 2., 3., 0., 4., 1.])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").cumprod(skipna=False)
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  3.,  0.,  4., nan])
        Coordinates:
          * time     (time) datetime64[ns] 48B 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        """
        return self._reduce_without_squeeze_warn(duck_array_ops.cumprod, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)