import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like

    Create a 'marketcolors' dict that is structured as expected
    by mplfinance._styles code:
        up     = color for close >= open
        down   = color for close  < open
        edge   = color for edge of candlestick; if "inherit"
                 then edge color will be same as up or down.
        wick   = color for wick of candlestick; if "inherit"
                 then wick color will be same as up or down.
        alpha  = opacity, 0.0 to 1.0, of candlestick face.
        ohlc   = color of ohlc bars when all the same color;
                 if ohlc == "inherit" then use up/down colors.
        volume = color of volume bars when all the same color;
                 if volume == "inherit" then use up/down colors.
    