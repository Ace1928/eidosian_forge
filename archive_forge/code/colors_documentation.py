from .._lazyload import matplotlib as mpl
from . import tools
import numpy as np
Create a discrete colormap with an arbitrary number of colors.

    This colormap chooses the best of the following, in order:
    - `plt.cm.tab10`
    - `plt.cm.tab20`
    - `scprep.plot.colors.tab30`
    - `scprep.plot.colors.tab40`
    - `scprep.plot.colors.tab10_continuous`

    If the number of colors required is less than the number of colors
    available, colors are selected specifically in order to reduce similarity
    between selected colors.

    Parameters
    ----------
    n : int, optional (default: 10)
        Number of required colors.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    