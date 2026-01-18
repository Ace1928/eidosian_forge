import numpy as np
import matplotlib as mpl
from matplotlib import _api, cm, patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
def _gen_starting_points(shape):
    """
    Yield starting points for streamlines.

    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    ny, nx = shape
    xfirst = 0
    yfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    x, y = (0, 0)
    direction = 'right'
    for i in range(nx * ny):
        yield (x, y)
        if direction == 'right':
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = 'up'
        elif direction == 'up':
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = 'left'
        elif direction == 'left':
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = 'down'
        elif direction == 'down':
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = 'right'