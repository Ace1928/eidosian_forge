import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
def _maybe_split_collections(do_split):
    if not do_split:
        return
    for fig in map(plt.figure, plt.get_fignums()):
        for ax in fig.axes:
            for coll in ax.collections:
                if isinstance(coll, mpl.contour.ContourSet):
                    with pytest.warns(mpl._api.MatplotlibDeprecationWarning):
                        coll.collections