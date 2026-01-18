from node A to node B means that the (qu)bit passes from the output of A
from collections import OrderedDict, defaultdict, deque, namedtuple
import copy
import math
from typing import Dict, Generator, Any, List
import numpy as np
import rustworkx as rx
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources, node_resources, CONTROL_FLOW_OP_NAMES
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit.bit import Bit
def _check_condition(self, name, condition):
    """Verify that the condition is valid.

        Args:
            name (string): used for error reporting
            condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)

        Raises:
            DAGCircuitError: if conditioning on an invalid register
        """
    if condition is None:
        return
    resources = condition_resources(condition)
    for creg in resources.cregs:
        if creg.name not in self.cregs:
            raise DAGCircuitError(f'invalid creg in condition for {name}')
    if not set(resources.clbits).issubset(self.clbits):
        raise DAGCircuitError(f'invalid clbits in condition for {name}')