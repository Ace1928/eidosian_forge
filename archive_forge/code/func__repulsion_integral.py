import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def _repulsion_integral(*args):
    """Compute the electron-electron repulsion integral for four contracted Gaussian functions.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the electron repulsion integral between four contracted Gaussian functions
        """
    args_a = [arg[0] for arg in args]
    args_b = [arg[1] for arg in args]
    args_c = [arg[2] for arg in args]
    args_d = [arg[3] for arg in args]
    alpha, ca, ra = _generate_params(basis_a.params, args_a)
    beta, cb, rb = _generate_params(basis_b.params, args_b)
    gamma, cc, rc = _generate_params(basis_c.params, args_c)
    delta, cd, rd = _generate_params(basis_d.params, args_d)
    if basis_a.params[1].requires_grad or normalize:
        ca = ca * primitive_norm(basis_a.l, alpha)
        cb = cb * primitive_norm(basis_b.l, beta)
        cc = cc * primitive_norm(basis_c.l, gamma)
        cd = cd * primitive_norm(basis_d.l, delta)
        n1 = contracted_norm(basis_a.l, alpha, ca)
        n2 = contracted_norm(basis_b.l, beta, cb)
        n3 = contracted_norm(basis_c.l, gamma, cc)
        n4 = contracted_norm(basis_d.l, delta, cd)
    else:
        n1 = n2 = n3 = n4 = 1.0
    e = n1 * n2 * n3 * n4 * (ca * cb[:, np.newaxis] * cc[:, np.newaxis, np.newaxis] * cd[:, np.newaxis, np.newaxis, np.newaxis] * electron_repulsion(basis_a.l, basis_b.l, basis_c.l, basis_d.l, ra, rb, rc, rd, alpha, beta[:, np.newaxis], gamma[:, np.newaxis, np.newaxis], delta[:, np.newaxis, np.newaxis, np.newaxis])).sum()
    return e