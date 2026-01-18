import warnings
from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires
class AmplitudeAndPhaseAndFreq:
    """Class storing combined amplitude, phase and freq callables"""

    def __init__(self, trig_fn, amp, phase, freq, hz_to_rads=2 * np.pi):
        self.amp_is_callable = callable(amp)
        self.phase_is_callable = callable(phase)
        self.freq_is_callable = callable(freq)
        if self.amp_is_callable and self.phase_is_callable and self.freq_is_callable:

            def callable_amp_and_phase_and_freq(params, t):
                return hz_to_rads * amp(params[0], t) * trig_fn(phase(params[1], t) + hz_to_rads * freq(params[2], t) * t)
            self.func = callable_amp_and_phase_and_freq
            return
        if self.amp_is_callable and self.phase_is_callable:

            def callable_amp_and_phase(params, t):
                return hz_to_rads * amp(params[0], t) * trig_fn(phase(params[1], t) + hz_to_rads * freq * t)
            self.func = callable_amp_and_phase
            return
        if self.amp_is_callable and self.freq_is_callable:

            def callable_amp_and_freq(params, t):
                return hz_to_rads * amp(params[0], t) * trig_fn(phase + hz_to_rads * freq(params[1], t) * t)
            self.func = callable_amp_and_freq
            return
        if self.phase_is_callable and self.freq_is_callable:

            def callable_phase_and_freq(params, t):
                return hz_to_rads * amp * trig_fn(phase(params[0], t) + hz_to_rads * freq(params[1], t) * t)
            self.func = callable_phase_and_freq
            return
        if self.amp_is_callable:

            def callable_amp(params, t):
                return hz_to_rads * amp(params[0], t) * trig_fn(phase + hz_to_rads * freq * t)
            self.func = callable_amp
            return
        if self.phase_is_callable:

            def callable_phase(params, t):
                return hz_to_rads * amp * trig_fn(phase(params[0], t) + hz_to_rads * freq * t)
            self.func = callable_phase
            return
        if self.freq_is_callable:

            def callable_freq(params, t):
                return hz_to_rads * amp * trig_fn(phase + hz_to_rads * freq(params[0], t) * t)
            self.func = callable_freq
            return

        def no_callable(_, t):
            return hz_to_rads * amp * trig_fn(phase + hz_to_rads * freq * t)
        self.func = no_callable

    def __call__(self, params, t):
        return self.func(params, t)