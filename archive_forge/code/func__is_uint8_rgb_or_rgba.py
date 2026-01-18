import datetime
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import numpy as np
def _is_uint8_rgb_or_rgba(tup):
    """ Deterine if rgb or rgba is in (0-255) format:
    Matplotlib expects rgb (and rgba) tuples to contain
    three (or four) floats between 0.0 and 1.0 
    
    Some people express rgb as tuples of three integers
    between 0 and 255.
    (In rgba, alpha is still a float from 0.0 to 1.0)
    """
    if isinstance(tup, str):
        return False
    if not np.iterable(tup):
        return False
    L = len(tup)
    if L < 3 or L > 4:
        return False
    if L == 4 and (tup[3] < 0 or tup[3] > 1):
        return False
    return not any([not isinstance(v, (int, np.unsignedinteger)) or v < 0 or v > 255 for v in tup[0:3]])