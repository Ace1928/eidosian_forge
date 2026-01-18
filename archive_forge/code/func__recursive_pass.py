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
def _recursive_pass(self, initial_layout):
    """Get a new instance of this class to handle a recursive call for a control-flow block.

        Each pass starts with its own new seed, determined deterministically from our own."""
    return self.__class__(self.coupling_map, trials=self.trials, seed=self._new_seed(), fake_run=self.fake_run, initial_layout=initial_layout)