import datetime
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import numpy as np
def _mpf_is_color_like(c):
    """Determine if an object is a color.
    
    Identical to `matplotlib.colors.is_color_like()`
    BUT ALSO considers int (0-255) rgb and rgba colors.
    """
    if mcolors.is_color_like(c):
        return True
    return _is_uint8_rgb_or_rgba(c)