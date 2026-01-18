import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _moment_integral(*args):
    """Normalize and compute the multipole moment integral for two contracted Gaussians.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the multipole moment integral between two contracted Gaussian orbitals
        """
    args_a = [arg[0] for arg in args]
    args_b = [arg[1] for arg in args]
    la = basis_a.l
    lb = basis_b.l
    alpha, ca, ra = _generate_params(basis_a.params, args_a)
    beta, cb, rb = _generate_params(basis_b.params, args_b)
    if basis_a.params[1].requires_grad or normalize:
        ca = ca * primitive_norm(basis_a.l, alpha)
        cb = cb * primitive_norm(basis_b.l, beta)
        na = contracted_norm(basis_a.l, alpha, ca)
        nb = contracted_norm(basis_b.l, beta, cb)
    else:
        na = nb = 1.0
    p = alpha[:, np.newaxis] + beta
    q = qml.math.sqrt(np.pi / p)
    r = (alpha[:, np.newaxis] * ra[:, np.newaxis, np.newaxis] + beta * rb[:, np.newaxis, np.newaxis]) / p
    i, j, k = qml.math.roll(qml.math.array([0, 2, 1]), idx)
    s = gaussian_moment(la[i], lb[i], ra[i], rb[i], alpha[:, np.newaxis], beta, order, r[i]) * expansion(la[j], lb[j], ra[j], rb[j], alpha[:, np.newaxis], beta, 0) * q * expansion(la[k], lb[k], ra[k], rb[k], alpha[:, np.newaxis], beta, 0) * q
    return (na * nb * (ca[:, np.newaxis] * cb) * s).sum()