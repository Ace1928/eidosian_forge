from typing import Callable, Sequence, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2
def _verify_measurements(circuits):
    """Verify the circuit has appropriate measurements."""
    for circuit in circuits:
        has_measurement = any((isinstance(op.gate, cirq.MeasurementGate) for moment in circuit for op in moment))
        if not has_measurement:
            raise RuntimeError('Code must measure at least one qubit.')