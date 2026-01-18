import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
def _show_cmap(cmap):
    """Show a continuous matplotlib colormap."""
    from .rcmod import axes_style
    with axes_style('white'):
        f, ax = plt.subplots(figsize=(8.25, 0.75))
    ax.set(xticks=[], yticks=[])
    x = np.linspace(0, 1, 256)[np.newaxis, :]
    ax.pcolormesh(x, cmap=cmap)