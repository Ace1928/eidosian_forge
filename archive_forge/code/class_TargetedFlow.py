import collections
import functools
from taskflow import deciders as de
from taskflow import exceptions as exc
from taskflow import flow
from taskflow.types import graph as gr
class TargetedFlow(Flow):
    """Graph flow with a target.

    Adds possibility to execute a flow up to certain graph node
    (task or subflow).
    """

    def __init__(self, *args, **kwargs):
        super(TargetedFlow, self).__init__(*args, **kwargs)
        self._subgraph = None
        self._target = None

    def set_target(self, target_node):
        """Set target for the flow.

        Any node(s) (tasks or subflows) not needed for the target
        node will not be executed.
        """
        if not self._graph.has_node(target_node):
            raise ValueError("Node '%s' not found" % target_node)
        self._target = target_node
        self._subgraph = None

    def reset_target(self):
        """Reset target for the flow.

        All node(s) of the flow will be executed.
        """
        self._target = None
        self._subgraph = None
    add = _reset_cached_subgraph(Flow.add)
    link = _reset_cached_subgraph(Flow.link)

    def _get_subgraph(self):
        if self._subgraph is not None:
            return self._subgraph
        if self._target is None:
            return self._graph
        nodes = [self._target]
        nodes.extend(self._graph.bfs_predecessors_iter(self._target))
        self._subgraph = gr.DiGraph(incoming_graph_data=self._graph.subgraph(nodes))
        self._subgraph.freeze()
        return self._subgraph