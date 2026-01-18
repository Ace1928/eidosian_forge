import matplotlib.pyplot as plt
import matplotlib.figure as mplfigure
import matplotlib.axes   as mpl_axes
from   mplfinance import _styles
import numpy as np
def _check_for_and_apply_style(kwargs):
    if 'style' in kwargs:
        style = kwargs['style']
        del kwargs['style']
    else:
        style = 'default'
    if not _styles._valid_mpf_style(style):
        raise TypeError('Invalid mplfinance style')
    if isinstance(style, str):
        style = _styles._get_mpfstyle(style)
    if isinstance(style, dict):
        _styles._apply_mpfstyle(style)
    else:
        raise TypeError('style should be a `dict`; why is it not?')
    return style