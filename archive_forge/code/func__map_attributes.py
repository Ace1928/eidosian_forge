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
def _map_attributes(self, arg, levels, defaults, attr):
    """Handle the specification for a given style attribute."""
    if arg is True:
        lookup_table = dict(zip(levels, defaults))
    elif isinstance(arg, dict):
        missing = set(levels) - set(arg)
        if missing:
            err = f'These `{attr}` levels are missing values: {missing}'
            raise ValueError(err)
        lookup_table = arg
    elif isinstance(arg, Sequence):
        arg = self._check_list_length(levels, arg, attr)
        lookup_table = dict(zip(levels, arg))
    elif arg:
        err = f'This `{attr}` argument was not understood: {arg}'
        raise ValueError(err)
    else:
        lookup_table = {}
    return lookup_table