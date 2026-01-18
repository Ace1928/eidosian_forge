import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def _clear_schema(self):
    self._labels = []
    self._properties = []
    self._relationship_types = []