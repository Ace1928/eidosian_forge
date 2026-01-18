import threading
import fasteners
from oslo_utils import excutils
from taskflow import flow
from taskflow import logging
from taskflow import task
from taskflow.types import graph as gr
from taskflow.types import tree as tr
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.flow import (LINK_INVARIANT, LINK_RETRY)  # noqa
class Compilation(object):
    """The result of a compilers ``compile()`` is this *immutable* object."""
    TASK = TASK
    RETRY = RETRY
    FLOW = FLOW
    '\n    Flow **entry** nodes will have a ``kind`` metadata key with\n    this value.\n    '
    FLOW_END = FLOW_END
    '\n    Flow **exit** nodes will have a ``kind`` metadata key with\n    this value (only applicable for compilation execution graph, not currently\n    used in tree hierarchy).\n    '

    def __init__(self, execution_graph, hierarchy):
        self._execution_graph = execution_graph
        self._hierarchy = hierarchy

    @property
    def execution_graph(self):
        """The execution ordering of atoms (as a graph structure)."""
        return self._execution_graph

    @property
    def hierarchy(self):
        """The hierarchy of patterns (as a tree structure)."""
        return self._hierarchy