import datetime
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import numpy as np
def _mpf_to_rgba(c, alpha=None):
    cnew = c
    if _is_uint8_rgb_or_rgba(c) and any((e > 1 for e in c[:3])):
        cnew = tuple([e / 255.0 for e in c[:3]])
        if len(c) == 4:
            cnew += c[3:]
    return mcolors.to_rgba(cnew, alpha)