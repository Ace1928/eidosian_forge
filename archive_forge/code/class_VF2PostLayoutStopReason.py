from enum import Enum
import logging
import inspect
import itertools
import time
from rustworkx import PyDiGraph, vf2_mapping, PyGraph
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.passes.layout import vf2_utils
class VF2PostLayoutStopReason(Enum):
    """Stop reasons for VF2PostLayout pass."""
    SOLUTION_FOUND = 'solution found'
    NO_BETTER_SOLUTION_FOUND = 'no better solution found'
    NO_SOLUTION_FOUND = 'nonexistent solution'
    MORE_THAN_2Q = '>2q gates in basis'