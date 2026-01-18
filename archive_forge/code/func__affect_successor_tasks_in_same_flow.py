import abc
import itertools
from taskflow import deciders
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states
def _affect_successor_tasks_in_same_flow(atom, runtime):
    execution_graph = runtime.compilation.execution_graph
    successors_iter = traversal.depth_first_iterate(execution_graph, atom, traversal.Direction.FORWARD, through_flows=False, through_retries=False)
    runtime.reset_atoms(itertools.chain([atom], successors_iter), state=states.IGNORE, intention=states.IGNORE)