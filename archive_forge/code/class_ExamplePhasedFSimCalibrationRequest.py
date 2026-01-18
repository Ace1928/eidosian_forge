from typing import Iterable, Optional, Tuple
import collections
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq_google
from cirq_google.calibration.engine_simulator import (
from cirq_google.calibration import (
import cirq
class ExamplePhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):

    def to_calibration_layer(self) -> cirq_google.CalibrationLayer:
        return NotImplemented

    def parse_result(self, result: cirq_google.CalibrationResult, job: Optional[cirq_google.EngineJob]=None) -> PhasedFSimCalibrationResult:
        return NotImplemented