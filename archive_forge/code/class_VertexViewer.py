import math
import os
from typing import Any, Mapping, Sequence
from langchain_core.runnables.graph import Edge as LangEdge
class VertexViewer:
    """Class to define vertex box boundaries that will be accounted for during
    graph building by grandalf.

    Args:
        name (str): name of the vertex.
    """
    HEIGHT = 3

    def __init__(self, name: str) -> None:
        self._h = self.HEIGHT
        self._w = len(name) + 2

    @property
    def h(self) -> int:
        """Height of the box."""
        return self._h

    @property
    def w(self) -> int:
        """Width of the box."""
        return self._w