from itertools import product
import numpy as np
from .utils import format_nvec
def _adjust_spine_placement(ax):
    """Helper function to set some common axis properties when plotting."""
    ax.xaxis.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)