import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
def _add_pad(self, x_min, x_max, y_min, y_max):
    """Perform the padding mentioned in `__call__`."""
    dx = (x_max - x_min) / self.nx
    dy = (y_max - y_min) / self.ny
    return (x_min - dx, x_max + dx, y_min - dy, y_max + dy)