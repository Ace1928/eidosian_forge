import pytest
import cirq
class MeasurementKeysGate(cirq.Gate):

    def _measurement_key_names_(self):
        return frozenset(['a', 'b'])

    def _measurement_key_objs_(self):
        return frozenset([cirq.MeasurementKey('c'), cirq.MeasurementKey('d')])

    def num_qubits(self) -> int:
        return 1