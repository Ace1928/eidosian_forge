import dataclasses
from typing import Callable, cast, Collection, Dict, Iterator, Optional, Sequence, Type, Union
import numpy as np
import sympy
import cirq
from cirq.devices import line_qubit
from cirq.ops import common_gates, parity_gates
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
def _serialize_measurement_gate(self, gate: cirq.MeasurementGate, targets: Sequence[int]) -> dict:
    key = cirq.measurement_key_name(gate)
    if chr(31) in key or chr(30) in key:
        raise ValueError(f'Measurement gates for IonQ API cannot have a key with a ascii unitor record separator in it. Key was {key}')
    return {'gate': 'meas', 'key': key, 'targets': ','.join((str(t) for t in targets))}