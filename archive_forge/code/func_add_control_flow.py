from io import StringIO
from warnings import warn
from shutil import get_terminal_size
import collections
import sys
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit import ControlledGate, Reset, Measure
from qiskit.circuit import ControlFlowOp, WhileLoopOp, IfElseOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.classical import expr
from qiskit.circuit.controlflow import node_resources
from qiskit.circuit.library.standard_gates import IGate, RZZGate, SwapGate, SXGate, SXdgGate
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from ._utils import (
from ..exceptions import VisualizationError
def add_control_flow(self, node, layers, wire_map):
    """Add control flow ops to the circuit drawing."""
    if isinstance(node.op, SwitchCaseOp) and isinstance(node.op.target, expr.Expr) or (getattr(node.op, 'condition', None) and isinstance(node.op.condition, expr.Expr)):
        condition = node.op.target if isinstance(node.op, SwitchCaseOp) else node.op.condition
        if self._builder is None:
            self._builder = QASM3Builder(self._circuit, includeslist=('stdgates.inc',), basis_gates=('U',), disable_constants=False, allow_aliasing=False)
            self._builder.build_classical_declarations()
        stream = StringIO()
        BasicPrinter(stream, indent='  ').visit(self._builder.build_expression(condition))
        self._expr_text = stream.getvalue()
        if len(self._expr_text) > self.expr_len:
            self._expr_text = self._expr_text[:self.expr_len] + '...'
    flow_layer = self.draw_flow_box(node, wire_map, CF_LEFT)
    layers.append(flow_layer.full_layer)
    circuit_list = list(node.op.blocks)
    if isinstance(node.op, SwitchCaseOp):
        circuit_list.insert(0, list(node.op.cases_specifier())[0][1].copy_empty_like())
    for circ_num, circuit in enumerate(circuit_list):
        flow_wire_map = wire_map.copy()
        flow_wire_map.update({inner: wire_map[outer] for outer, inner in zip(node.qargs, circuit.qubits)})
        for outer, inner in zip(node.cargs, circuit.clbits):
            if self.cregbundle and (in_reg := get_bit_register(self._circuit, inner)) is not None:
                out_reg = get_bit_register(self._circuit, outer)
                flow_wire_map.update({in_reg: wire_map[out_reg]})
            else:
                flow_wire_map.update({inner: wire_map[outer]})
        if circ_num > 0:
            flow_layer = self.draw_flow_box(node, flow_wire_map, CF_MID, circ_num - 1)
            layers.append(flow_layer.full_layer)
        _, _, nodes = _get_layered_instructions(circuit, wire_map=flow_wire_map)
        for layer_nodes in nodes:
            flow_layer2 = Layer(self.qubits, self.clbits, self.cregbundle, self._circuit, flow_wire_map)
            for layer_node in layer_nodes:
                if isinstance(layer_node.op, ControlFlowOp):
                    self._nest_depth += 1
                    self.add_control_flow(layer_node, layers, flow_wire_map)
                    self._nest_depth -= 1
                else:
                    flow_layer2, current_cons, current_cons_cond, connection_label = self._node_to_gate(layer_node, flow_layer2, flow_wire_map)
                    flow_layer2.connections.append((connection_label, current_cons))
                    flow_layer2.connections.append((None, current_cons_cond))
            flow_layer2.connect_with('│')
            layers.append(flow_layer2.full_layer)
    flow_layer = self.draw_flow_box(node, flow_wire_map, CF_RIGHT)
    layers.append(flow_layer.full_layer)