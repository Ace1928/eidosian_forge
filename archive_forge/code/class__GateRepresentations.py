from typing import (
import re
import warnings
from dataclasses import dataclass
import cirq
from cirq_google import ops
from cirq_google import transformers
from cirq_google.api import v2
from cirq_google.devices import known_devices
from cirq_google.experimental import ops as experimental_ops
@dataclass
class _GateRepresentations:
    """Contains equivalent representations of a gate in both DeviceSpecification and GridDevice.

    Attributes:
        gate_spec_name: The name of gate type in `GateSpecification`.
        deserialized_forms: Gate representations to be included when the corresponding
            `GateSpecification` gate type is deserialized into gatesets and gate durations.
        serializable_forms: GateFamilies used to check whether a given gate can be serialized to the
            gate type in this _GateRepresentation.
    """
    gate_spec_name: str
    deserialized_forms: List[GateOrFamily]
    serializable_forms: List[cirq.GateFamily]