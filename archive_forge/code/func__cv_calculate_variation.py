import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt):
    """Returns the variation of level set 'phi' based on algorithm parameters.

    This corresponds to equation (22) of the paper by Pascal Getreuer,
    which computes the next iteration of the level set based on a current
    level set.

    A full explanation regarding all the terms is beyond the scope of the
    present description, but there is one difference of particular import.
    In the original algorithm, convergence is accelerated, and required
    memory is reduced, by using a single array. This array, therefore, is a
    combination of non-updated and updated values. If this were to be
    implemented in python, this would require a double loop, where the
    benefits of having fewer iterations would be outweided by massively
    increasing the time required to perform each individual iteration. A
    similar approach is used by Rami Cohen, and it is from there that the
    C1-4 notation is taken.
    """
    eta = 1e-16
    P = np.pad(phi, 1, mode='edge')
    phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
    phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
    phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
    phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    C1 = 1.0 / np.sqrt(eta + phixp ** 2 + phiy0 ** 2)
    C2 = 1.0 / np.sqrt(eta + phixn ** 2 + phiy0 ** 2)
    C3 = 1.0 / np.sqrt(eta + phix0 ** 2 + phiyp ** 2)
    C4 = 1.0 / np.sqrt(eta + phix0 ** 2 + phiyn ** 2)
    K = P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 + P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4
    Hphi = (phi > 0).astype(image.dtype)
    c1, c2 = _cv_calculate_averages(image, Hphi)
    difference_from_average_term = -lambda1 * (image - c1) ** 2 + lambda2 * (image - c2) ** 2
    new_phi = phi + dt * _cv_delta(phi) * (mu * K + difference_from_average_term)
    return new_phi / (1 + mu * dt * _cv_delta(phi) * (C1 + C2 + C3 + C4))