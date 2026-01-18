import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
@value.value_equality
class _InverseCompositeGate(Gate):
    """The inverse of a composite gate."""

    def __init__(self, original: Gate) -> None:
        self._original = original

    def _qid_shape_(self):
        return protocols.qid_shape(self._original)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self._original
        return NotImplemented

    def _decompose_(self, qubits):
        return self._decompose_with_context_(qubits)

    def _decompose_with_context_(self, qubits: Sequence['cirq.Qid'], context: Optional['cirq.DecompositionContext']=None) -> 'cirq.OP_TREE':
        return protocols.inverse(protocols.decompose_once_with_qubits(self._original, qubits, context=context))

    def _has_unitary_(self):
        from cirq import protocols, devices
        qubits = devices.LineQid.for_gate(self)
        return all((protocols.has_unitary(op) for op in protocols.decompose_once_with_qubits(self._original, qubits)))

    @cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._original)

    @cached_method
    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._original)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> '_InverseCompositeGate':
        return _InverseCompositeGate(protocols.resolve_parameters(self._original, resolver, recursive))

    def _value_equality_values_(self):
        return self._original

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        sub_info = protocols.circuit_diagram_info(self._original, args, default=NotImplemented)
        if sub_info is NotImplemented:
            return NotImplemented
        sub_info.exponent *= -1
        return sub_info

    def __repr__(self) -> str:
        return f'({self._original!r}**-1)'