import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def _refresh_relations(self):
    rels = self.relationship_types()
    self._relationship_types = [r[0] for _, r in enumerate(rels)]