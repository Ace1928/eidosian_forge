from __future__ import annotations
import typing
from itertools import cycle, islice
import numpy as np
import pandas as pd
from ..coords import coord_flip
from ..scales.scale_discrete import scale_discrete
from .annotate import annotate
from .geom import geom
from .geom_polygon import geom_polygon
from .geom_rect import geom_rect

        Draw stripes on every panel
        