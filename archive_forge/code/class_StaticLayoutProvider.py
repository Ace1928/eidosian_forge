from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
class StaticLayoutProvider(LayoutProvider):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    graph_layout = Dict(Either(Int, String), Len(Seq(Float), 2), default={}, help='\n    The coordinates of the graph nodes in cartesian space. The keys of\n    the dictionary correspond to node indices or labels and the values\n    are two element sequences containing the x and y coordinates of\n    the nodes.\n\n    .. code-block:: python\n\n        {\n            0 : [0.5, 0.5],\n            1 : [1.0, 0.86],\n            2 : [0.86, 1],\n        }\n    ')