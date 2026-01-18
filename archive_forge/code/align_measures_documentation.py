from __future__ import annotations
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
from qiskit.circuit.quantumcircuit import ClbitSpecifier, QubitSpecifier
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.deprecation import deprecate_func
Pad idle time-slots in ``qubits`` with delays in ``unit`` until ``until``.