import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def _enforceRGB(c):
    tc = toColor(c)
    if isinstance(tc, CMYKColor):
        if tc.cyan == tc.magenta == tc.yellow == 0:
            v = 1 - tc.black * tc.density
            tc = Color(v, v, v, alpha=tc.alpha)
        else:
            _enforceError('RGB', c, tc)
    return tc