from __future__ import annotations
import typing
from functools import partial
from collections.abc import Callable
from typing import Protocol
import numpy as np
from qiskit.quantum_info import Operator
from .approximate import ApproximateCircuit, ApproximatingObjective

        Approximately compiles a circuit represented as a unitary matrix by solving an optimization
        problem defined by ``approximating_objective`` and using ``approximate_circuit`` as a
        template for the approximate circuit.

        Args:
            target_matrix: a unitary matrix to approximate.
            approximate_circuit: a template circuit that will be filled with the parameter values
                obtained in the optimization procedure.
            approximating_objective: a definition of the optimization problem.
            initial_point: initial values of angles/parameters to start optimization from.
        