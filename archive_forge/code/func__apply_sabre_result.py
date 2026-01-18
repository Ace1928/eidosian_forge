import logging
from copy import deepcopy
import time
import rustworkx
from qiskit.circuit import SwitchCaseOp, ControlFlowOp, Clbit, ClassicalRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.dagcircuit import DAGCircuit
from qiskit.utils.parallel import CPU_COUNT
from qiskit._accelerate.sabre_swap import (
from qiskit._accelerate.nlayout import NLayout
def _apply_sabre_result(out_dag, in_dag, sabre_result, initial_layout, physical_qubits, circuit_to_dag_dict):
    """Apply the ``SabreResult`` to ``out_dag``, mutating it in place.  This function in effect
    performs the :class:`.ApplyLayout` transpiler pass with ``initial_layout`` and the Sabre routing
    simultaneously, though it assumes that ``out_dag`` has already been prepared as containing the
    right physical qubits.

    Mutates ``out_dag`` in place and returns it.  Mutates ``initial_layout`` in place as scratch
    space.

    Args:
        out_dag (DAGCircuit): the physical DAG that the output should be written to.
        in_dag (DAGCircuit): the source of the nodes that are being routed.
        sabre_result (tuple[SwapMap, Sequence[int], NodeBlockResults]): the result object from the
            Rust run of the Sabre routing algorithm.
        initial_layout (NLayout): a Rust-space mapping of virtual indices (i.e. those of the qubits
            in ``in_dag``) to physical ones.
        physical_qubits (list[Qubit]): an indexable sequence of :class:`.circuit.Qubit` objects
            representing the physical qubits of the circuit.  Note that disjoint-coupling
            handling can mean that these are not strictly a "canonical physical register" in order.
        circuit_to_dag_dict (Mapping[int, DAGCircuit]): a mapping of the Python object identity
            (as returned by :func:`id`) of a control-flow block :class:`.QuantumCircuit` to a
            :class:`.DAGCircuit` that represents the same thing.
    """
    swap_singleton = SwapGate()

    def empty_dag(block):
        empty = DAGCircuit()
        empty.add_qubits(out_dag.qubits)
        for qreg in out_dag.qregs.values():
            empty.add_qreg(qreg)
        empty.add_clbits(block.clbits)
        for creg in block.cregs:
            empty.add_creg(creg)
        empty._global_phase = block.global_phase
        return empty

    def apply_swaps(dest_dag, swaps, layout):
        for a, b in swaps:
            qubits = (physical_qubits[a], physical_qubits[b])
            layout.swap_physical(a, b)
            dest_dag.apply_operation_back(swap_singleton, qubits, (), check=False)

    def recurse(dest_dag, source_dag, result, root_logical_map, layout):
        """The main recursive worker.  Mutates ``dest_dag`` and ``layout`` and returns them.

        ``root_virtual_map`` is a mapping of the (virtual) qubit in ``source_dag`` to the index of
        the virtual qubit in the root source DAG that it is bound to."""
        swap_map, node_order, node_block_results = result
        for node_id in node_order:
            node = source_dag._multi_graph[node_id]
            if node_id in swap_map:
                apply_swaps(dest_dag, swap_map[node_id], layout)
            if not isinstance(node.op, ControlFlowOp):
                dest_dag.apply_operation_back(node.op, [physical_qubits[layout.virtual_to_physical(root_logical_map[q])] for q in node.qargs], node.cargs, check=False)
                continue
            block_results = node_block_results[node_id]
            mapped_block_dags = []
            idle_qubits = set(dest_dag.qubits)
            for block, block_result in zip(node.op.blocks, block_results):
                block_root_logical_map = {inner: root_logical_map[outer] for inner, outer in zip(block.qubits, node.qargs)}
                block_dag, block_layout = recurse(empty_dag(block), circuit_to_dag_dict[id(block)], (block_result.result.map, block_result.result.node_order, block_result.result.node_block_results), block_root_logical_map, layout.copy())
                apply_swaps(block_dag, block_result.swap_epilogue, block_layout)
                mapped_block_dags.append(block_dag)
                idle_qubits.intersection_update(block_dag.idle_wires())
            mapped_blocks = []
            for mapped_block_dag in mapped_block_dags:
                mapped_block_dag.remove_qubits(*idle_qubits)
                mapped_blocks.append(dag_to_circuit(mapped_block_dag))
            mapped_node = node.op.replace_blocks(mapped_blocks)
            mapped_node_qargs = mapped_blocks[0].qubits if mapped_blocks else ()
            dest_dag.apply_operation_back(mapped_node, mapped_node_qargs, node.cargs, check=False)
        return (dest_dag, layout)
    root_logical_map = {bit: index for index, bit in enumerate(in_dag.qubits)}
    return recurse(out_dag, in_dag, sabre_result, root_logical_map, initial_layout)[0]