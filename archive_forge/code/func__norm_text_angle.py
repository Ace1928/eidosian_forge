import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def _norm_text_angle(a):
    """Return the given angle normalized to -90 < *a* <= 90 degrees."""
    a = (a + 180) % 180
    if a > 90:
        a = a - 180
    return a