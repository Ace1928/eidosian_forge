import pandas as pd
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
def _scale_width_config(scale, width_config):
    if scale['volume'] is not None:
        width_config['volume_width'] *= scale['volume']
    if scale['ohlc'] is not None:
        width_config['ohlc_ticksize'] *= scale['ohlc']
    if scale['candle'] is not None:
        width_config['candle_width'] *= scale['candle']
    if scale['lines'] is not None:
        width_config['line_width'] *= scale['lines']
    if scale['volume_linewidth'] is not None:
        width_config['volume_linewidth'] *= scale['volume_linewidth']
    if scale['ohlc_linewidth'] is not None:
        width_config['ohlc_linewidth'] *= scale['ohlc_linewidth']
    if scale['candle_linewidth'] is not None:
        width_config['candle_linewidth'] *= scale['candle_linewidth']