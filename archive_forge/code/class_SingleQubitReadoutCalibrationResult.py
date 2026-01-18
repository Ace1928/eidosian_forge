import dataclasses
import time
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
import sympy
import numpy as np
from cirq import circuits, ops, study
@dataclasses.dataclass
class SingleQubitReadoutCalibrationResult:
    """Result of estimating single qubit readout error.

    Attributes:
        zero_state_errors: A dictionary from qubit to probability of measuring
            a 1 when the qubit is initialized to |0⟩.
        one_state_errors: A dictionary from qubit to probability of measuring
            a 0 when the qubit is initialized to |1⟩.
        repetitions: The number of repetitions that were used to estimate the
            probabilities.
        timestamp: The time the data was taken, in seconds since the epoch.
    """
    zero_state_errors: Dict['cirq.Qid', float]
    one_state_errors: Dict['cirq.Qid', float]
    repetitions: int
    timestamp: float

    def _json_dict_(self) -> Dict[str, Any]:
        return {'zero_state_errors': list(self.zero_state_errors.items()), 'one_state_errors': list(self.one_state_errors.items()), 'repetitions': self.repetitions, 'timestamp': self.timestamp}

    @classmethod
    def _from_json_dict_(cls, zero_state_errors, one_state_errors, repetitions, timestamp, **kwargs):
        return cls(zero_state_errors=dict(zero_state_errors), one_state_errors=dict(one_state_errors), repetitions=repetitions, timestamp=timestamp)

    def __repr__(self) -> str:
        return f'cirq.experiments.SingleQubitReadoutCalibrationResult(zero_state_errors={self.zero_state_errors!r}, one_state_errors={self.one_state_errors!r}, repetitions={self.repetitions!r}, timestamp={self.timestamp!r})'