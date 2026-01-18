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
def _pnf_calculator(indf, boxsize, reverse=3, method='hilo'):
    """Calculate Point and Figure Numbers

    TODO: Support arbitrary column names of OHLC
    """

    def round_to(n, precision):
        correction = 0.5 if n >= 0 else -0.5
        return int(n / precision + correction) * precision
    if method == 'hilo':
        Xprices = indf.High
        Oprices = indf.Low
    elif method == 'open':
        Xprices = indf.Open
        Oprices = indf.Open
    elif method == 'close':
        Xprices = indf.Close
        Oprices = indf.Close
    else:
        raise ValueError('Bad value for method="' + str(method) + '"')
    indf.loc[:, 'XBox'] = [int(x / boxsize) * boxsize for x in Xprices]
    indf.loc[:, 'OBox'] = [int(round_to(x + 0.5 * boxsize, boxsize) / boxsize) * boxsize for x in Oprices]
    v = indf.iloc[2].Close - indf.iloc[0].Open
    xo = 'X' if v > 0 else 'O'
    d0 = indf.index[0]
    if xo == 'X':
        column = [indf.OBox[d0] - boxsize]
    else:
        column = [indf.XBox[d0] + boxsize]
    xo
    pnf = {}
    pnf[d0] = column
    column_count = 1
    current_column = pnf[d0]
    for d in indf.index[1:]:
        current_level = current_column[-1]
        new_column = []
        if xo == 'X':
            box = indf.XBox[d]
            reverse = current_level - 3 * boxsize
            if box > current_level:
                num = int(round((box - current_level) / boxsize))
                for jj in range(1, num + 1):
                    current_column.append(current_level + jj * boxsize)
            elif indf.OBox[d] <= reverse:
                top = current_level - boxsize
                box = indf.OBox[d]
                num = int(round((top - box) / boxsize))
                new_column = [top]
                for jj in range(1, num + 1):
                    new_column.append(top - jj * boxsize)
                pnf[d] = new_column
                xo = 'O'
                current_column = new_column
                column_count += 1
        else:
            box = indf.OBox[d]
            reverse = current_level + 3 * boxsize
            if round_to(box, 1.0 * boxsize) < current_level:
                num = int(round((current_level - box) / boxsize))
                for jj in range(1, num + 1):
                    current_column.append(current_level - jj * boxsize)
            elif indf.XBox[d] >= reverse:
                bot = current_level + boxsize
                box = indf.XBox[d]
                num = int(round((box - bot) / boxsize))
                new_column = [bot]
                for jj in range(1, num + 1):
                    new_column.append(bot + jj * boxsize)
                pnf[d] = new_column
                xo = 'X'
                current_column = new_column
                column_count += 1
    return pnf