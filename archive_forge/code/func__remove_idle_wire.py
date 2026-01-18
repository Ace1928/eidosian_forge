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
def _remove_idle_wire(self, wire):
    """Remove an idle qubit or bit from the circuit.

        Args:
            wire (Bit): the wire to be removed, which MUST be idle.
        """
    inp_node = self.input_map[wire]
    oup_node = self.output_map[wire]
    self._multi_graph.remove_node(inp_node._node_id)
    self._multi_graph.remove_node(oup_node._node_id)
    self._wires.remove(wire)
    del self.input_map[wire]
    del self.output_map[wire]