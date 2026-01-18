from __future__ import annotations
import random
from typing import TYPE_CHECKING
from matplotlib import patches
import matplotlib.lines as mlines
import numpy as np
from pandas.core.dtypes.missing import notna
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
def _get_marker_compat(marker):
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker