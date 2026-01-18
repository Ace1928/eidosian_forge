from itertools import cycle
import numpy as np
from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll
@staticmethod
def _create_patch(orig_handle, xdescent, ydescent, width, height):
    return Rectangle(xy=(-xdescent, -ydescent), width=width, height=height, color=orig_handle.get_facecolor())