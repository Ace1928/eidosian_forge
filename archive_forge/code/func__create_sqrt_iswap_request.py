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
def _create_sqrt_iswap_request(pairs: Iterable[Tuple[cirq.Qid, cirq.Qid]], options: FloquetPhasedFSimCalibrationOptions=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION) -> FloquetPhasedFSimCalibrationRequest:
    return FloquetPhasedFSimCalibrationRequest(gate=cirq.FSimGate(np.pi / 4, 0.0), pairs=tuple(pairs), options=options)