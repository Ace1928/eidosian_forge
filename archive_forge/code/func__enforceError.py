import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def _enforceError(kind, c, tc):
    if isinstance(tc, Color):
        xtra = tc._lookupName()
        xtra = xtra and '(%s)' % xtra or ''
    else:
        xtra = ''
    raise ValueError('Non %s color %r%s' % (kind, c, xtra))