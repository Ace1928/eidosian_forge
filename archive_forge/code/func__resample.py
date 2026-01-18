from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
def _resample(self, resample_cls: type[T_Resample], indexer: Mapping[Any, str] | None, skipna: bool | None, closed: SideOptions | None, label: SideOptions | None, base: int | None, offset: pd.Timedelta | datetime.timedelta | str | None, origin: str | DatetimeLike, loffset: datetime.timedelta | str | None, restore_coord_dims: bool | None, **indexer_kwargs: str) -> T_Resample:
    """Returns a Resample object for performing resampling operations.

        Handles both downsampling and upsampling. The resampled
        dimension must be a datetime-like coordinate. If any intervals
        contain no values from the original object, they will be given
        the value ``NaN``.

        Parameters
        ----------
        indexer : {dim: freq}, optional
            Mapping from the dimension name to resample frequency [1]_. The
            dimension must be datetime-like.
        skipna : bool, optional
            Whether to skip missing values when aggregating in downsampling.
        closed : {"left", "right"}, optional
            Side of each interval to treat as closed.
        label : {"left", "right"}, optional
            Side of each interval to use for labeling.
        base : int, optional
            For frequencies that evenly subdivide 1 day, the "origin" of the
            aggregated intervals. For example, for "24H" frequency, base could
            range from 0 through 23.

            .. deprecated:: 2023.03.0
                Following pandas, the ``base`` parameter is deprecated in favor
                of the ``origin`` and ``offset`` parameters, and will be removed
                in a future version of xarray.

        origin : {'epoch', 'start', 'start_day', 'end', 'end_day'}, pd.Timestamp, datetime.datetime, np.datetime64, or cftime.datetime, default 'start_day'
            The datetime on which to adjust the grouping. The timezone of origin
            must match the timezone of the index.

            If a datetime is not used, these values are also supported:
            - 'epoch': `origin` is 1970-01-01
            - 'start': `origin` is the first value of the timeseries
            - 'start_day': `origin` is the first day at midnight of the timeseries
            - 'end': `origin` is the last value of the timeseries
            - 'end_day': `origin` is the ceiling midnight of the last day
        offset : pd.Timedelta, datetime.timedelta, or str, default is None
            An offset timedelta added to the origin.
        loffset : timedelta or str, optional
            Offset used to adjust the resampled time labels. Some pandas date
            offset strings are supported.

            .. deprecated:: 2023.03.0
                Following pandas, the ``loffset`` parameter is deprecated in favor
                of using time offset arithmetic, and will be removed in a future
                version of xarray.

        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        **indexer_kwargs : {dim: freq}
            The keyword arguments form of ``indexer``.
            One of indexer or indexer_kwargs must be provided.

        Returns
        -------
        resampled : same type as caller
            This object resampled.

        Examples
        --------
        Downsample monthly time-series data to seasonal data:

        >>> da = xr.DataArray(
        ...     np.linspace(0, 11, num=12),
        ...     coords=[
        ...         pd.date_range(
        ...             "1999-12-15",
        ...             periods=12,
        ...             freq=pd.DateOffset(months=1),
        ...         )
        ...     ],
        ...     dims="time",
        ... )
        >>> da
        <xarray.DataArray (time: 12)> Size: 96B
        array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])
        Coordinates:
          * time     (time) datetime64[ns] 96B 1999-12-15 2000-01-15 ... 2000-11-15
        >>> da.resample(time="QS-DEC").mean()
        <xarray.DataArray (time: 4)> Size: 32B
        array([ 1.,  4.,  7., 10.])
        Coordinates:
          * time     (time) datetime64[ns] 32B 1999-12-01 2000-03-01 ... 2000-09-01

        Upsample monthly time-series data to daily data:

        >>> da.resample(time="1D").interpolate("linear")  # +doctest: ELLIPSIS
        <xarray.DataArray (time: 337)> Size: 3kB
        array([ 0.        ,  0.03225806,  0.06451613,  0.09677419,  0.12903226,
                0.16129032,  0.19354839,  0.22580645,  0.25806452,  0.29032258,
                0.32258065,  0.35483871,  0.38709677,  0.41935484,  0.4516129 ,
                0.48387097,  0.51612903,  0.5483871 ,  0.58064516,  0.61290323,
                0.64516129,  0.67741935,  0.70967742,  0.74193548,  0.77419355,
                0.80645161,  0.83870968,  0.87096774,  0.90322581,  0.93548387,
                0.96774194,  1.        ,  1.03225806,  1.06451613,  1.09677419,
                1.12903226,  1.16129032,  1.19354839,  1.22580645,  1.25806452,
                1.29032258,  1.32258065,  1.35483871,  1.38709677,  1.41935484,
                1.4516129 ,  1.48387097,  1.51612903,  1.5483871 ,  1.58064516,
                1.61290323,  1.64516129,  1.67741935,  1.70967742,  1.74193548,
                1.77419355,  1.80645161,  1.83870968,  1.87096774,  1.90322581,
                1.93548387,  1.96774194,  2.        ,  2.03448276,  2.06896552,
                2.10344828,  2.13793103,  2.17241379,  2.20689655,  2.24137931,
                2.27586207,  2.31034483,  2.34482759,  2.37931034,  2.4137931 ,
                2.44827586,  2.48275862,  2.51724138,  2.55172414,  2.5862069 ,
                2.62068966,  2.65517241,  2.68965517,  2.72413793,  2.75862069,
                2.79310345,  2.82758621,  2.86206897,  2.89655172,  2.93103448,
                2.96551724,  3.        ,  3.03225806,  3.06451613,  3.09677419,
                3.12903226,  3.16129032,  3.19354839,  3.22580645,  3.25806452,
        ...
                7.87096774,  7.90322581,  7.93548387,  7.96774194,  8.        ,
                8.03225806,  8.06451613,  8.09677419,  8.12903226,  8.16129032,
                8.19354839,  8.22580645,  8.25806452,  8.29032258,  8.32258065,
                8.35483871,  8.38709677,  8.41935484,  8.4516129 ,  8.48387097,
                8.51612903,  8.5483871 ,  8.58064516,  8.61290323,  8.64516129,
                8.67741935,  8.70967742,  8.74193548,  8.77419355,  8.80645161,
                8.83870968,  8.87096774,  8.90322581,  8.93548387,  8.96774194,
                9.        ,  9.03333333,  9.06666667,  9.1       ,  9.13333333,
                9.16666667,  9.2       ,  9.23333333,  9.26666667,  9.3       ,
                9.33333333,  9.36666667,  9.4       ,  9.43333333,  9.46666667,
                9.5       ,  9.53333333,  9.56666667,  9.6       ,  9.63333333,
                9.66666667,  9.7       ,  9.73333333,  9.76666667,  9.8       ,
                9.83333333,  9.86666667,  9.9       ,  9.93333333,  9.96666667,
               10.        , 10.03225806, 10.06451613, 10.09677419, 10.12903226,
               10.16129032, 10.19354839, 10.22580645, 10.25806452, 10.29032258,
               10.32258065, 10.35483871, 10.38709677, 10.41935484, 10.4516129 ,
               10.48387097, 10.51612903, 10.5483871 , 10.58064516, 10.61290323,
               10.64516129, 10.67741935, 10.70967742, 10.74193548, 10.77419355,
               10.80645161, 10.83870968, 10.87096774, 10.90322581, 10.93548387,
               10.96774194, 11.        ])
        Coordinates:
          * time     (time) datetime64[ns] 3kB 1999-12-15 1999-12-16 ... 2000-11-15

        Limit scope of upsampling method

        >>> da.resample(time="1D").nearest(tolerance="1D")
        <xarray.DataArray (time: 337)> Size: 3kB
        array([ 0.,  0., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan,  1.,  1.,  1., nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan,  2.,  2.,  2., nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,  3.,
                3.,  3., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan,  4.,  4.,  4., nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan,  5.,  5.,  5., nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                6.,  6.,  6., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan,  7.,  7.,  7., nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan,  8.,  8.,  8., nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan,  9.,  9.,  9., nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, 10., 10., 10., nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
               nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 11., 11.])
        Coordinates:
          * time     (time) datetime64[ns] 3kB 1999-12-15 1999-12-16 ... 2000-11-15

        See Also
        --------
        pandas.Series.resample
        pandas.DataFrame.resample

        References
        ----------
        .. [1] https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects
        """
    from xarray.core.dataarray import DataArray
    from xarray.core.groupby import ResolvedGrouper, TimeResampler
    from xarray.core.resample import RESAMPLE_DIM
    if skipna is not None and (not isinstance(skipna, bool)) or ('how' in indexer_kwargs and 'how' not in self.dims) or ('dim' in indexer_kwargs and 'dim' not in self.dims):
        raise TypeError("resample() no longer supports the `how` or `dim` arguments. Instead call methods on resample objects, e.g., data.resample(time='1D').mean()")
    indexer = either_dict_or_kwargs(indexer, indexer_kwargs, 'resample')
    if len(indexer) != 1:
        raise ValueError('Resampling only supported along single dimensions.')
    dim, freq = next(iter(indexer.items()))
    dim_name: Hashable = dim
    dim_coord = self[dim]
    group = DataArray(dim_coord, coords=dim_coord.coords, dims=dim_coord.dims, name=RESAMPLE_DIM)
    grouper = TimeResampler(freq=freq, closed=closed, label=label, origin=origin, offset=offset, loffset=loffset, base=base)
    rgrouper = ResolvedGrouper(grouper, group, self)
    return resample_cls(self, (rgrouper,), dim=dim_name, resample_dim=RESAMPLE_DIM, restore_coord_dims=restore_coord_dims)