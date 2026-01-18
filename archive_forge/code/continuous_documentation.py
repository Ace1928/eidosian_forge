from __future__ import annotations
import functools
import numpy as np
from qiskit.pulse.exceptions import PulseError
Continuous Y-only correction DRAG pulse for standard nonlinear oscillator (SNO) [1].

    [1] Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
        Analytic control methods for high-fidelity unitary operations
        in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).

    Args:
        times: Times to output pulse for.
        amp: Pulse amplitude at `center`.
        center: Center (mean) of pulse.
        sigma: Width (standard deviation) of pulse.
        beta: Y correction amplitude. For the SNO this is $\beta=-\frac{\lambda_1^2}{4\Delta_2}$.
            Where $\lambds_1$ is the relative coupling strength between the first excited and second
            excited states and $\Delta_2$ is the detuning between the respective excited states.
        zeroed_width: Subtract baseline of drag pulse to make sure
            $\Omega_g(center \pm zeroed_width/2)=0$ is satisfied. This is used to avoid
            large discontinuities at the start of a drag pulse.
        rescale_amp: If `zeroed_width` is not `None` and `rescale_amp=True` the pulse will
            be rescaled so that $\Omega_g(center)=amp$.

    