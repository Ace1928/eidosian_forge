import itertools
from typing import Optional
from unittest import mock
import numpy as np
import pytest
import cirq
import cirq_google
import cirq_google.calibration.workflow as workflow
import cirq_google.calibration.xeb_wrapper
from cirq.experiments import (
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
def _fsim_identity_converter(gate: cirq.Gate) -> Optional[PhaseCalibratedFSimGate]:
    if isinstance(gate, cirq.FSimGate):
        return PhaseCalibratedFSimGate(gate, 0.0)
    return None