import pandas as pd
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
def _get_widths_df():
    """
    Provide a dataframe of width data that appropriate scales widths of
    various aspects of the plot (candles,ohlc bars,volume bars) based on
    the amount or density of data.  These numbers were arrived at by 
    carefully testing many use-cases of plots with various styles, 
    and observing which numbers gave the "best" appearance.
    """
    numpoints = [n for n in range(30, 241, 30)]
    volume_width = (0.98, 0.96, 0.95, 0.925, 0.9, 0.9, 0.875, 0.825)
    volume_linewidth = tuple([0.65] * 8)
    candle_width = (0.65, 0.575, 0.5, 0.445, 0.435, 0.425, 0.42, 0.415)
    candle_linewidth = (1.0, 0.875, 0.75, 0.625, 0.5, 0.438, 0.435, 0.435)
    ohlc_tickwidth = tuple([0.35] * 8)
    ohlc_linewidth = (1.5, 1.175, 0.85, 0.525, 0.525, 0.525, 0.525, 0.525)
    line_width = (2.25, 1.8, 1.3, 0.813, 0.807, 0.801, 0.796, 0.791)
    widths = {}
    widths['vw'] = volume_width
    widths['vlw'] = volume_linewidth
    widths['cw'] = candle_width
    widths['clw'] = candle_linewidth
    widths['ow'] = ohlc_tickwidth
    widths['olw'] = ohlc_linewidth
    widths['lw'] = line_width
    return pd.DataFrame(widths, index=numpoints)