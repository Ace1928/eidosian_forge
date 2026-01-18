import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Sequence, cast, Dict, Any, List, Iterator
import cirq
from cirq import _compat, study
@dataclass(frozen=True)
class BitstringsMeasurement:
    """Use in-circuit MeasurementGate to collect many repetitions of strings of bits.

    This is the lowest-level measurement type allowed in `QuantumExecutable` and behaves
    identically to the `cirq.Sampler.run` function. The executable's circuit must contain
    explicit measurement gates.

    Args:
        n_repeitions: The number of repetitions to execute the circuit.
    """
    n_repetitions: int

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    def __repr__(self):
        return cirq._compat.dataclass_repr(self, namespace='cirq_google')