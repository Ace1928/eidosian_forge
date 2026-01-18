from functools import wraps
from inspect import signature
import warnings
import numpy as np
from autoray import numpy as anp
import pennylane as qml
def _reconstruct_equ(fun, num_frequency, x0=None, f0=None, interface=None):
    """Reconstruct a univariate Fourier series with consecutive integer
    frequencies, using trigonometric interpolation and equidistant shifts.

    This technique is based on
    `Dirichlet kernels <https://en.wikipedia.org/wiki/Dirichlet_kernel>`_, see
    `Vidal and Theis (2018) <https://arxiv.org/abs/1812.06323>`_ or
    `Wierichs et al. (2022) <https://doi.org/10.22331/q-2022-03-30-677>`_.

    Args:
        fun (callable): Univariate finite Fourier series to reconstruct.
            It must have signature ``float -> float`` .
        num_frequency (int): Number of integer frequencies in ``fun``.
            All integer frequencies below ``num_frequency`` are assumed
            to be present in ``fun`` as well; if they are not, the output
            is correct put the reconstruction could have been performed
            with fewer evaluations of ``fun`` .
        x0 (float): Center to which to shift the reconstruction.
            The points at which ``fun`` is evaluated are *not* affected
            by ``x0`` .
        f0 (float): Value of ``fun`` at zero; Providing ``f0`` saves one
            evaluation of ``fun``.
        interface (str): Which auto-differentiation framework to use as
            interface. This determines in which interface the output
            reconstructed function is intended to be used.

    Returns:
        callable: Reconstructed Fourier series with ``num_frequency`` frequencies.
        This function is a purely classical function. Furthermore, it is fully
        differentiable.
    """
    if not abs(int(num_frequency)) == num_frequency:
        raise ValueError(f'num_frequency must be a non-negative integer, got {num_frequency}')
    a = (num_frequency + 0.5) / np.pi
    b = 0.5 / np.pi
    shifts_pos = qml.math.arange(1, num_frequency + 1) / a
    shifts_neg = -shifts_pos[::-1]
    shifts = qml.math.concatenate([shifts_neg, [0.0], shifts_pos])
    shifts = anp.asarray(shifts, like=interface)
    f0 = fun(0.0) if f0 is None else f0
    evals = list(map(fun, shifts[:num_frequency])) + [f0] + list(map(fun, shifts[num_frequency + 1:]))
    evals = anp.asarray(evals, like=interface)
    x0 = anp.asarray(np.float64(0.0), like=interface) if x0 is None else x0

    def _reconstruction(x):
        """Univariate reconstruction based on equidistant shifts and Dirichlet kernels.
        The derivative at of ``sinc`` are not well-implemented in TensorFlow and Autograd,
        use the Fourier transform reconstruction if this derivative is needed.
        """
        _x = x - x0 - shifts
        return qml.math.tensordot(qml.math.sinc(a * _x) / qml.math.sinc(b * _x), evals, axes=[[0], [0]])
    return _reconstruction