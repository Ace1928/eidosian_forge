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
def _construct_pnf_scatter(ax, ptype, dates, xdates, opens, highs, lows, closes, volumes, config, style):
    """Represent the price change with Xs and Os

    NOTE: this code assumes if any value open, low, high, close is
    missing they all are missing

    Algorithm Explanation
    ---------------------
    In the first part of the algorithm ...

    Useful sources:
    https://...
    https://...

    Parameters
    ----------
    dates : sequence
        sequence of dates
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    config_pointnfig_params : kwargs table (dictionary)
        box_size : size of each box
        atr_length : length of time used for calculating atr
    closes : sequence
        sequence of closing values
    marketcolors : dict of colors: up, down, edge, wick, alpha

    Returns
    -------
    calculate_values : dict of assorted point-and-figure box calculation results
    """
    df = pd.DataFrame(dict(Open=opens, High=highs, Low=lows, Close=closes, Volume=volumes))
    if config['tz_localize']:
        df.index = pd.DatetimeIndex(mdates.num2date(dates)).tz_localize(None)
    else:
        df.index = pd.DatetimeIndex(mdates.num2date(dates))
    df.index.name = 'Date'
    marketcolors = style['marketcolors']
    if marketcolors is None:
        marketcolors = _get_mpfstyle('classic')['marketcolors']
    pointnfig_params = _process_kwargs(config['pnf_params'], _valid_pnf_kwargs())
    box_size = pointnfig_params['box_size']
    atr_length = pointnfig_params['atr_length']
    reversal = pointnfig_params['reversal']
    method = pointnfig_params['method']
    if box_size == 'atr':
        if atr_length == 'total':
            box_size = _calculate_atr(len(closes) - 1, df.High, df.Low, df.Close)
        else:
            box_size = _calculate_atr(atr_length, df.High, df.Low, df.Close)
    elif isinstance(box_size, str) and box_size[-1] == '%':
        percent = float(box_size[:-1])
        if not (percent > 0 and percent < 50):
            raise ValueError('Specified percent (for box_size) must be > 0. and < 50.')
        box_size = percent / 100.0 * df.Close.iloc[-1]
    else:
        upper_limit = (max(df.Close) - min(df.Close)) / 2
        lower_limit = 0.01 * _calculate_atr(len(df.Close) - 1, df.High, df.Low, df.Close)
        if box_size > upper_limit:
            raise ValueError('Specified box_size may not be larger than [50% of the close' + ' price range of the dataset] which has value: ' + str(upper_limit))
        elif box_size < lower_limit:
            raise ValueError('Specified box_size may not be smaller than [0.01* the Average' + ' True Value of the dataset) which has value: ' + str(lower_limit))
    if reversal < 1 or reversal > 9:
        raise ValueError('Specified reversal must be an integer in the range [1,9]')
    pnfd = _pnf_calculator(df, boxsize=box_size, reverse=reversal, method=method)
    yvals = [y for key in pnfd.keys() for y in pnfd[key]]
    ylim_top = max(yvals) + 0.5 * box_size
    ylim_bot = min(yvals) - 0.5 * box_size
    dpi = ax.figure.get_dpi()
    wxt = ax.get_window_extent()
    axis_height_inches = wxt.height / dpi
    max_vertical_boxes = (ylim_top - ylim_bot) / box_size
    inches_per_box = axis_height_inches / max_vertical_boxes
    ideal_marker_size = (inches_per_box * 72) ** 2
    ideal_marker_size *= 0.6
    marker_size = ideal_marker_size * pointnfig_params['scale_markers']
    alpha = marketcolors['alpha']
    if pointnfig_params['use_candle_colors']:
        uc = mcolors.to_rgba(marketcolors['candle']['up'], alpha)
        ue = mcolors.to_rgba(marketcolors['edge']['up'])
        uw = 0.5
        dc = mcolors.to_rgba(marketcolors['candle']['down'], alpha)
        de = mcolors.to_rgba(marketcolors['edge']['up'])
        dw = 0.5
    else:
        uc = mcolors.to_rgba(marketcolors['edge']['up'], alpha)
        ue = mcolors.to_rgba(marketcolors['edge']['up'], alpha)
        uw = 0.5
        dc = mcolors.to_rgba(marketcolors['candle']['down'], 0.0)
        de = mcolors.to_rgba(marketcolors['candle']['down'], alpha)
        dw = 0.18 * marker_size ** 0.5
    xvals = []
    yvals = []
    mvals = []
    cvals = []
    evals = []
    lwids = []
    jj = 0
    for key in pnfd.keys():
        m = 'X' if pnfd[key][0] < pnfd[key][-1] else 'o'
        c = uc if pnfd[key][0] < pnfd[key][-1] else dc
        e = ue if pnfd[key][0] < pnfd[key][-1] else de
        w = uw if pnfd[key][0] < pnfd[key][-1] else dw
        for v in pnfd[key]:
            yvals.append(v)
            xvals.append(jj)
            mvals.append(m)
            evals.append(e)
            cvals.append(c)
            lwids.append(w)
        jj += 1
    plot_yvals = [y + 0.5 * box_size for y in yvals]
    _ = _mscatter(xvals, plot_yvals, ax, mvals, s=marker_size, c=cvals, linewidths=lwids, edgecolors=evals)
    if config['volume'] is not None:
        pnf_volumes = []
        d1list = [d for d in pnfd.keys()]
        d2list = d1list[1:] + [df.index[-1]]
        for d1, d2 in zip(d1list, d2list):
            pnf_volumes.append(df.Volume.loc[d1:d2].sum())
    else:
        pnf_volumes = [0] * len(xvals)
    hi = max(yvals)
    lo = min(yvals)
    xlen = int(round((hi - lo) / box_size, 0) + 2)
    pad = (xlen - xvals[-1]) * pointnfig_params['scale_right_padding']
    pad = max(0, pad)
    xdates = np.arange(len(pnfd) + int(pad))
    pnf_volumes = pnf_volumes + [float('nan')] * int(pad)
    pnf_results = dict(pnf_volumes=pnf_volumes, pnf_ylimits=(ylim_bot, ylim_top), pnf_values=pnfd, pnf_df=df, pnf_boxsize=box_size, pnf_xdates=xdates)
    return pnf_results