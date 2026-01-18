from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
class NodesOnly(GraphHitTestPolicy):
    """
    With the ``NodesOnly`` policy, only graph nodes are able to be selected and
    inspected. There is no selection or inspection of graph edges.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)