import abc
import itertools
from taskflow import deciders
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states
def _affect_atom(atom, runtime):
    runtime.reset_atoms([atom], state=states.IGNORE, intention=states.IGNORE)