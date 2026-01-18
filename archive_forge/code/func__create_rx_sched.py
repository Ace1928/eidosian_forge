from typing import Union
from functools import lru_cache
import numpy as np
from qiskit.circuit import Instruction
from qiskit.pulse import Schedule, ScheduleBlock, builder, ScalableSymbolicPulse
from qiskit.pulse.channels import Channel
from qiskit.pulse.library.symbolic_pulses import Drag
from qiskit.transpiler.passes.calibration.base_builder import CalibrationBuilder
from qiskit.transpiler import Target
from qiskit.circuit.library.standard_gates import RXGate
from qiskit.exceptions import QiskitError
@lru_cache
def _create_rx_sched(rx_angle: float, duration: int, amp: float, sigma: float, beta: float, channel: Channel):
    """Generates (and caches) pulse calibrations for RX gates.
    Assumes that the rotation angle is in [0, pi].
    """
    new_amp = rx_angle / (np.pi / 2) * amp
    with builder.build() as new_rx_sched:
        builder.play(Drag(duration=duration, amp=new_amp, sigma=sigma, beta=beta, angle=0), channel=channel)
    return new_rx_sched