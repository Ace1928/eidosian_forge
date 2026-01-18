from __future__ import annotations
import abc
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable
from inspect import signature
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.base_tasks import GenericPass, PassManagerIR
from qiskit.passmanager.compilation_status import PropertySet, RunState, PassManagerState
from .exceptions import TranspilerError
from .layout import TranspileLayout
Runs the pass on circuit.

        Args:
            circuit: The dag on which the pass is run.
            property_set: Input/output property set. An analysis pass
                might change the property set in-place.

        Returns:
            If on transformation pass, the resulting QuantumCircuit.
            If analysis pass, the input circuit.
        