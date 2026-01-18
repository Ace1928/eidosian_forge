import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def _enforceSEP(c):
    """pure separating colors only, this makes black a problem"""
    tc = toColor(c)
    if not isinstance(tc, CMYKColorSep):
        _enforceError('separating', c, tc)
    return tc