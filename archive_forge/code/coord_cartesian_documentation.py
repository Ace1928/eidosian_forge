from __future__ import annotations
import typing
from types import SimpleNamespace
from ..iapi import panel_view
from ..positions.position import transform_position
from .coord import coord, dist_euclidean

        Compute the range and break information for the panel
        