import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def _enforceSEP_BLACK(c):
    """separating + blacks only"""
    tc = toColor(c)
    if not isinstance(tc, CMYKColorSep):
        if isinstance(tc, Color) and tc.red == tc.blue == tc.green:
            tc = _CMYK_black.clone(density=1 - tc.red)
        elif not (isinstance(tc, CMYKColor) and tc.cyan == tc.magenta == tc.yellow == 0):
            _enforceError('separating or black', c, tc)
    return tc