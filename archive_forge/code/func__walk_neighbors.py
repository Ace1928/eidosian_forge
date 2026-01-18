import abc
import itertools
from taskflow import deciders
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states
def _walk_neighbors():
    execution_graph = runtime.compilation.execution_graph
    for node in execution_graph.successors(atom):
        node_data = execution_graph.nodes[node]
        if node_data['kind'] == compiler.TASK:
            yield node