from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
from typing import TYPE_CHECKING
def _get_bins_and_eval(self, data, orient, groupby, scale_type):
    bin_kws = self._define_bin_params(data, orient, scale_type)
    return groupby.apply(data, self._eval, orient, bin_kws)