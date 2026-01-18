from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
class EdgesAndLinkedNodes(GraphHitTestPolicy):
    """
    With the ``EdgesAndLinkedNodes`` policy, inspection or selection of graph
    edges will result in the inspection or selection of the edge and of the
    linked graph nodes. There is no direct selection or inspection of graph
    nodes.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)