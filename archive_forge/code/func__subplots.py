import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def _subplots():
    _, axs = plt.subplots(rows, cols)
    axs = axs.flat
    for ax, color in zip(axs, colors):
        ax.plot(x, y, color=color)
        add_offsetboxes(ax, 20, color=color)
    return axs