from __future__ import annotations
from collections.abc import Iterable
import logging
from qiskit.circuit import Qubit, Clbit, Instruction
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
Interleave instruction sequence in between two nodes.

        .. note::
            If a DAGOpNode is added here, it should update node_start_time property
            in the property set so that the added node is also scheduled.
            This is achieved by adding operation via :meth:`_apply_scheduled_op`.

        .. note::

            This method doesn't check if the total duration of new DAGOpNode added here
            is identical to the interval (``t_end - t_start``).
            A developer of the pass must guarantee this is satisfied.
            If the duration is greater than the interval, your circuit may be
            compiled down to the target code with extra duration on the backend compiler,
            which is then played normally without error. However, the outcome of your circuit
            might be unexpected due to erroneous scheduling.

        Args:
            dag: DAG circuit that sequence is applied.
            qubit: The wire that the sequence is applied on.
            t_start: Absolute start time of this interval.
            t_end: Absolute end time of this interval.
            next_node: Node that follows the sequence.
            prev_node: Node ahead of the sequence.
        