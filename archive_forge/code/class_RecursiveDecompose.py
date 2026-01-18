import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class RecursiveDecompose(cirq.Gate):

    def __init__(self, recurse: bool=True, mock_qm=mock.Mock(spec=cirq.QubitManager), with_context: bool=False):
        self.recurse = recurse
        self.mock_qm = mock_qm
        self.with_context = with_context

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_impl(self, qubits, mock_qm: mock.Mock):
        mock_qm.qalloc(self.recurse)
        yield (RecursiveDecompose(recurse=False, mock_qm=self.mock_qm, with_context=self.with_context).on(*qubits) if self.recurse else cirq.Z.on_each(*qubits))
        mock_qm.qfree(self.recurse)

    def _decompose_(self, qubits):
        if self.with_context:
            assert False
        else:
            return self._decompose_impl(qubits, self.mock_qm)

    def _decompose_with_context_(self, qubits, context):
        if self.with_context:
            qm = self.mock_qm if context is None else context.qubit_manager
            return self._decompose_impl(qubits, qm)
        else:
            return NotImplemented

    def _has_unitary_(self):
        return True