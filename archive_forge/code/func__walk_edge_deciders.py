import collections
import functools
from futurist import waiters
from taskflow import deciders as de
from taskflow.engines.action_engine.actions import retry as ra
from taskflow.engines.action_engine.actions import task as ta
from taskflow.engines.action_engine import builder as bu
from taskflow.engines.action_engine import compiler as com
from taskflow.engines.action_engine import completer as co
from taskflow.engines.action_engine import scheduler as sched
from taskflow.engines.action_engine import scopes as sc
from taskflow.engines.action_engine import selector as se
from taskflow.engines.action_engine import traversal as tr
from taskflow import exceptions as exc
from taskflow import logging
from taskflow import states as st
from taskflow.utils import misc
from taskflow.flow import (LINK_DECIDER, LINK_DECIDER_DEPTH)  # noqa
def _walk_edge_deciders(self, graph, atom):
    """Iterates through all nodes, deciders that alter atoms execution."""
    predecessors_iter = graph.predecessors
    nodes = collections.deque(((u_node, atom) for u_node in predecessors_iter(atom)))
    visited = set()
    while nodes:
        u_node, v_node = nodes.popleft()
        u_node_kind = graph.nodes[u_node]['kind']
        u_v_data = graph.adj[u_node][v_node]
        try:
            decider = u_v_data[LINK_DECIDER]
            decider_depth = u_v_data.get(LINK_DECIDER_DEPTH)
            if decider_depth is None:
                decider_depth = de.Depth.ALL
            yield _EdgeDecider(u_node, u_node_kind, decider, decider_depth)
        except KeyError:
            pass
        if u_node_kind == com.FLOW and u_node not in visited:
            visited.add(u_node)
            nodes.extend(((u_u_node, u_node) for u_u_node in predecessors_iter(u_node)))