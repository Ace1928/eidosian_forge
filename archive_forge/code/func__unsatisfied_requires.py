import collections
import functools
from taskflow import deciders as de
from taskflow import exceptions as exc
from taskflow import flow
from taskflow.types import graph as gr
def _unsatisfied_requires(node, graph, *additional_provided):
    requires = set(node.requires)
    if not requires:
        return requires
    for provided in additional_provided:
        requires = requires.difference(provided)
        if not requires:
            return requires
    for pred in graph.bfs_predecessors_iter(node):
        requires = requires.difference(pred.provides)
        if not requires:
            return requires
    return requires