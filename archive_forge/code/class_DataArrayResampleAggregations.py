from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from xarray.core import duck_array_ops
from xarray.core.options import OPTIONS
from xarray.core.types import Dims, Self
from xarray.core.utils import contains_only_chunked_or_numpy, module_available
class DataArrayResampleAggregations:
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").count()
        <xarray.DataArray (time: 3)> Size: 24B
        array([1, 3, 1])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").all()
        <xarray.DataArray (time: 3)> Size: 3B
        array([ True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").any()
        <xarray.DataArray (time: 3)> Size: 3B
        array([ True,  True,  True])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").max()
        <xarray.DataArray (time: 3)> Size: 24B
        array([1., 3., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").max(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([ 1.,  3., nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").min()
        <xarray.DataArray (time: 3)> Size: 24B
        array([1., 0., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").min(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([ 1.,  0., nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").mean()
        <xarray.DataArray (time: 3)> Size: 24B
        array([1.        , 1.66666667, 2.        ])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").mean(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([1.        , 1.66666667,        nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").prod()
        <xarray.DataArray (time: 3)> Size: 24B
        array([1., 0., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").prod(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([ 1.,  0., nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.resample(time="3ME").prod(skipna=True, min_count=2)
        <xarray.DataArray (time: 3)> Size: 24B
        array([nan,  0., nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").sum()
        <xarray.DataArray (time: 3)> Size: 24B
        array([1., 5., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").sum(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([ 1.,  5., nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.resample(time="3ME").sum(skipna=True, min_count=2)
        <xarray.DataArray (time: 3)> Size: 24B
        array([nan,  5., nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").std()
        <xarray.DataArray (time: 3)> Size: 24B
        array([0.        , 1.24721913, 0.        ])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").std(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([0.        , 1.24721913,        nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.resample(time="3ME").std(skipna=True, ddof=1)
        <xarray.DataArray (time: 3)> Size: 24B
        array([       nan, 1.52752523,        nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").var()
        <xarray.DataArray (time: 3)> Size: 24B
        array([0.        , 1.55555556, 0.        ])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").var(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([0.        , 1.55555556,        nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.resample(time="3ME").var(skipna=True, ddof=1)
        <xarray.DataArray (time: 3)> Size: 24B
        array([       nan, 2.33333333,        nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
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
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").median()
        <xarray.DataArray (time: 3)> Size: 24B
        array([1., 2., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").median(skipna=False)
        <xarray.DataArray (time: 3)> Size: 24B
        array([ 1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 24B 2001-01-31 2001-04-30 2001-07-31
        """
        return self._reduce_without_squeeze_warn(duck_array_ops.median, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

    def cumsum(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``cumsum`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``cumsum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").cumsum()
        <xarray.DataArray (time: 6)> Size: 48B
        array([1., 2., 5., 5., 2., 2.])
        Coordinates:
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Dimensions without coordinates: time

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").cumsum(skipna=False)
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  5.,  5.,  2., nan])
        Coordinates:
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Dimensions without coordinates: time
        """
        return self._reduce_without_squeeze_warn(duck_array_ops.cumsum, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)

    def cumprod(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None, **kwargs: Any) -> DataArray:
        """
        Reduce this DataArray's data by applying ``cumprod`` along some dimension(s).

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``cumprod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over the Resample dimensions.
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
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Use the ``flox`` package to significantly speed up resampling computations,
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

        >>> da.resample(time="3ME").cumprod()
        <xarray.DataArray (time: 6)> Size: 48B
        array([1., 2., 6., 0., 2., 2.])
        Coordinates:
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Dimensions without coordinates: time

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3ME").cumprod(skipna=False)
        <xarray.DataArray (time: 6)> Size: 48B
        array([ 1.,  2.,  6.,  0.,  2., nan])
        Coordinates:
            labels   (time) <U1 24B 'a' 'b' 'c' 'c' 'b' 'a'
        Dimensions without coordinates: time
        """
        return self._reduce_without_squeeze_warn(duck_array_ops.cumprod, dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)