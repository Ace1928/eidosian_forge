from __future__ import annotations
import typing
import warnings
import numpy as np
import pandas as pd
from .._utils import log
from ..coords import coord_flip
from ..exceptions import PlotnineWarning
from ..scales.scale_continuous import scale_continuous as ScaleContinuous
from .annotate import annotate
from .geom_path import geom_path
from .geom_rug import geom_rug
def _minor(x: Sequence[Any], mid_idx: int) -> AnyArray:
    return np.hstack([x[1:mid_idx], x[mid_idx + 1:-1]])