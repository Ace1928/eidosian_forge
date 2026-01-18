from typing import Optional, List, Tuple
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.annotated_operation import AnnotatedOperation, _canonicalize_modifiers
from qiskit.circuit import EquivalenceLibrary, ControlledGate, Operation, ControlFlowOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
def _recursively_process_definitions(self, op: Operation) -> bool:
    """
        Recursively applies optimizations to op's definition (or to op.base_op's
        definition if op is an annotated operation).
        Returns True if did something.
        """
    if isinstance(op, AnnotatedOperation):
        return self._recursively_process_definitions(op.base_op)
    controlled_gate_open_ctrl = isinstance(op, ControlledGate) and op._open_ctrl
    if not controlled_gate_open_ctrl:
        inst_supported = self._target.instruction_supported(operation_name=op.name) if self._target is not None else op.name in self._device_insts
        if inst_supported or (self._equiv_lib is not None and self._equiv_lib.has_entry(op)):
            return False
    try:
        definition = op.definition
    except TypeError as err:
        raise TranspilerError(f'OptimizeAnnotated was unable to extract definition for {op.name}: {err}') from err
    except AttributeError:
        definition = None
    if definition is None:
        raise TranspilerError(f'OptimizeAnnotated was unable to optimize {op}.')
    definition_dag = circuit_to_dag(definition, copy_operations=False)
    definition_dag, opt = self._run_inner(definition_dag)
    if opt:
        op.definition = dag_to_circuit(definition_dag)
    return opt