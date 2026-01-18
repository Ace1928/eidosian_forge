from typing import AbstractSet, Any, Optional, Tuple
import numpy as np
import cirq
from cirq._compat import proper_repr
Inits CouplerPulse.

        Args:
            hold_time: Length of the 'plateau' part of the coupler trajectory.
            coupling_mhz: Target qubit-qubit coupling reached at the plateau.
            rise_time: Width of the rising (or falling) action of the trapezoidal pulse.
            padding_time: Symmetric padding around the coupler pulse.
            q0_detune_mhz: Detuning of the first qubit.
            q1_detune_mhz: Detuning of the second qubit.

        