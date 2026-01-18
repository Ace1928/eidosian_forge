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
@staticmethod
def _bits_in_operation(operation):
    """Return an iterable over the classical bits that are inherent to an instruction.  This
        includes a `condition`, or the `target` of a :class:`.ControlFlowOp`.

        Args:
            instruction: the :class:`~.circuit.Instruction` instance for a node.

        Returns:
            Iterable[Clbit]: the :class:`.Clbit`\\ s involved.
        """
    if (condition := getattr(operation, 'condition', None)) is not None:
        yield from condition_resources(condition).clbits
    if isinstance(operation, SwitchCaseOp):
        target = operation.target
        if isinstance(target, Clbit):
            yield target
        elif isinstance(target, ClassicalRegister):
            yield from target
        else:
            yield from node_resources(target).clbits