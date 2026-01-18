import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def _enforceCMYK(c):
    """cmyk outputs only (rgb greys converted)"""
    tc = toColor(c)
    if not isinstance(tc, CMYKColor):
        if isinstance(tc, Color) and tc.red == tc.blue == tc.green:
            tc = _CMYK_black.clone(black=1 - tc.red, alpha=tc.alpha)
        else:
            _enforceError('CMYK', c, tc)
    elif isinstance(tc, CMYKColorSep):
        tc = tc.clone()
        tc.__class__ = CMYKColor
    return tc