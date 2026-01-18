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
def _draw(geom: geom, axis: Literal['x', 'y'], tick_positions: tuple[AnyArray, AnyArray, AnyArray]):
    for position, length in zip(tick_positions, lengths):
        data = pd.DataFrame({axis: position, **_aesthetics})
        geom.draw_group(data, panel_params, coord, ax, length=length, **params)