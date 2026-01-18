import collections
import contextlib
import itertools
import threading
from automaton import runners
from concurrent import futures
import fasteners
import functools
import networkx as nx
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import process_executor
from taskflow.engines.action_engine import runtime
from taskflow.engines import base
from taskflow import exceptions as exc
from taskflow import logging
from taskflow import states
from taskflow import storage
from taskflow.types import failure
from taskflow.utils import misc
@staticmethod
def _check_compilation(compilation):
    """Performs post compilation validation/checks."""
    seen = set()
    dups = set()
    execution_graph = compilation.execution_graph
    for node, node_attrs in execution_graph.nodes(data=True):
        if node_attrs['kind'] in compiler.ATOMS:
            atom_name = node.name
            if atom_name in seen:
                dups.add(atom_name)
            else:
                seen.add(atom_name)
    if dups:
        raise exc.Duplicate('Atoms with duplicate names found: %s' % sorted(dups))
    return compilation