import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
@dataclasses.dataclass(frozen=True)
class PhasedFSimCalibrationRequest(abc.ABC):
    """Description of the request to characterize PhasedFSimGate.

    Attributes:
        pairs: Set of qubit pairs to characterize. A single qubit can appear on at most one pair in
            the set.
        gate: Gate to characterize for each qubit pair from pairs. This must be a supported gate
            which can be described cirq.PhasedFSim gate. This gate must be serialized by the
            cirq_google.SerializableGateSet used
    """
    pairs: Tuple[Tuple[cirq.Qid, cirq.Qid], ...]
    gate: cirq.Gate
    options: PhasedFSimCalibrationOptions

    @property
    @lru_cache_typesafe
    def qubit_to_pair(self) -> MutableMapping[cirq.Qid, Tuple[cirq.Qid, cirq.Qid]]:
        """Returns mapping from qubit to a qubit pair that it belongs to."""
        return collections.ChainMap(*({q: pair for q in pair} for pair in self.pairs))

    @abc.abstractmethod
    def to_calibration_layer(self) -> CalibrationLayer:
        """Encodes this characterization request in a CalibrationLayer object."""

    @abc.abstractmethod
    def parse_result(self, result: CalibrationResult, job: Optional[EngineJob]=None) -> PhasedFSimCalibrationResult:
        """Decodes the characterization result issued for this request."""