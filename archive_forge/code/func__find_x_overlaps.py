from __future__ import annotations
import typing
from copy import copy
import numpy as np
import pandas as pd
from .._utils import groupby_apply, pivot_apply
from ..exceptions import PlotnineError
from .position_dodge import position_dodge
def _find_x_overlaps(gdf):
    return pd.DataFrame({'n': find_x_overlaps(gdf)})