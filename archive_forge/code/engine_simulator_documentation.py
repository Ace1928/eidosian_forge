from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
Generates a gate with drift for a given gate.

        Args:
            a: The first qubit.
            b: The second qubit.
            gate_calibration: Reference gate together with a phase information.

        Returns:
            A modified gate that includes the drifts induced by internal state of the simulator.
        