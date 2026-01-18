import itertools
import logging
from math import inf
import numpy as np
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit.classical import expr, types
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.circuit import (
from qiskit._accelerate import stochastic_swap as stochastic_swap_rs
from qiskit._accelerate import nlayout
from qiskit.transpiler.passes.layout import disjoint_utils
from .utils import get_swap_map_dag
def _new_seed(self):
    """Get a seed for a new RNG instance."""
    return self.rng.integers(9223372036854775807)