from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
def _lookup_single(self, key, attr=None):
    """Get attribute(s) for a given data point."""
    if attr is None:
        value = self.lookup_table[key]
    else:
        value = self.lookup_table[key][attr]
    return value