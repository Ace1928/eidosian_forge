from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def _nodes_and_edges(self):
    return (len(self.nodes), sum((len(node.parents) for node in self.node_map.values())))