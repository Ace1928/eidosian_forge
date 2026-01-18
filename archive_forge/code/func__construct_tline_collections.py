import numpy  as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from itertools import cycle
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.patches     import Ellipse
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _alines_validator, _bypass_kwarg_validation
from mplfinance._arg_validators import _xlim_validator, _is_datelike
from mplfinance._styles         import _get_mpfstyle
from mplfinance._helpers        import _mpf_to_rgba
from six.moves import zip
from matplotlib.ticker import Formatter
def _construct_tline_collections(tlines, dtix, dates, opens, highs, lows, closes):
    """construct trend line collections

    Parameters
    ----------
    tlines : sequence
        sequences of pairs of date[time]s

        date[time] may be (a) pandas.to_datetime parseable string,
                          (b) pandas Timestamp, or
                          (c) python datetime.datetime or datetime.date

    tlines may also be a dict, containing
    the following keys:

        'tlines'     : the same as defined above: sequence of pairs of date[time]s
        'colors'     : colors for the above tlines
        'linestyle'  : line types for the above tlines
        'linewidths' : line widths for the above tlines

    dtix:  date index for the x-axis, used for converting the dates when
           x-values are 'evenly spaced integers' (as when skipping non-trading days)

    Returns
    -------
    ret : list
        lines collections
    """
    if tlines is None:
        return None
    if isinstance(tlines, dict):
        tconfig = _process_kwargs(tlines, _valid_lines_kwargs())
        tlines = tconfig['tlines']
    else:
        tconfig = _process_kwargs({}, _valid_lines_kwargs())
    tline_use = tconfig['tline_use']
    tline_method = tconfig['tline_method']
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes}, index=pd.DatetimeIndex(mdates.num2date(dates)))
    df.index = df.index.tz_localize(None)

    def _tline_point_to_point(dfslice, tline_use):
        p1 = dfslice.iloc[0]
        p2 = dfslice.iloc[-1]
        x1 = p1.name
        y1 = p1[tline_use].mean()
        x2 = p2.name
        y2 = p2[tline_use].mean()
        return ((x1, y1), (x2, y2))

    def _tline_lsq(dfslice, tline_use):
        """
        This closed-form linear least squares algorithm was taken from:
        https://mmas.github.io/least-squares-fitting-numpy-scipy
        """
        si = dfslice[tline_use].mean(axis=1)
        s = si.dropna()
        if len(s) < 2:
            err = 'NOT enough data for Least Squares'
            if len(si) > 2:
                err += ', due to presence of NaNs'
            raise ValueError(err)
        xs = mdates.date2num(s.index.to_pydatetime())
        ys = s.values
        a = np.vstack([xs, np.ones(len(xs))]).T
        m, b = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, ys))
        x1, x2 = (xs[0], xs[-1])
        y1 = m * x1 + b
        y2 = m * x2 + b
        x1, x2 = (mdates.num2date(x1).replace(tzinfo=None), mdates.num2date(x2).replace(tzinfo=None))
        return ((x1, y1), (x2, y2))
    if isinstance(tline_use, str):
        tline_use = [tline_use]
    tline_use = [u.lower() for u in tline_use]
    alines = []
    for d1, d2 in tlines:
        dfslice = df.loc[d1:d2]
        if len(dfslice) < 2:
            dfdr = '\ndf date range: [' + str(df.index[0]) + ' , ' + str(df.index[-1]) + ']'
            raise ValueError('\ntlines date pair (' + str(d1) + ',' + str(d2) + ') too close, or wrong order, or out of range!' + dfdr)
        if tline_method == 'least squares' or tline_method == 'least-squares':
            p1, p2 = _tline_lsq(dfslice, tline_use)
        elif tline_method == 'point-to-point':
            p1, p2 = _tline_point_to_point(dfslice, tline_use)
        else:
            raise ValueError('\nUnrecognized value for `tline_method` = "' + str(tline_method) + '"')
        alines.append((p1, p2))
    del tconfig['alines']
    alines = dict(alines=alines, **tconfig)
    alines['tlines'] = None
    return _construct_aline_collections(alines, dtix)