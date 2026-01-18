from __future__ import annotations
import functools
import numpy as np
from qiskit.pulse.exceptions import PulseError
def _fix_sech_width(sech_samples: np.ndarray, amp: complex, center: float, sigma: float, zeroed_width: float | None=None, rescale_amp: bool=False, ret_scale_factor: bool=False) -> np.ndarray | tuple[np.ndarray, float]:
    """Enforce that the supplied sech pulse is zeroed at a specific width.

    This is achieved by subtracting $\\Omega_g(center \\pm zeroed_width/2)$ from all samples.

    amp: Pulse amplitude at `center`.
    center: Center (mean) of pulse.
    sigma: Standard deviation of pulse.
    zeroed_width: Subtract baseline from sech pulses to make sure
        $\\Omega_g(center \\pm zeroed_width/2)=0$ is satisfied. This is used to avoid
        large discontinuities at the start of a sech pulse. If unsupplied,
        defaults to $2*(center + 1)$ such that $\\Omega_g(-1)=0$ and $\\Omega_g(2*(center + 1))=0$.
    rescale_amp: If True the pulse will be rescaled so that $\\Omega_g(center)=amp$.
    ret_scale_factor: Return amplitude scale factor.
    """
    if zeroed_width is None:
        zeroed_width = 2 * (center + 1)
    zero_offset = sech(np.array([zeroed_width / 2]), amp, 0, sigma)
    sech_samples -= zero_offset
    amp_scale_factor: complex | float | np.ndarray = 1.0
    if rescale_amp:
        amp_scale_factor = amp / (amp - zero_offset) if amp - zero_offset != 0 else 1.0
        sech_samples *= amp_scale_factor
    if ret_scale_factor:
        return (sech_samples, amp_scale_factor)
    return sech_samples