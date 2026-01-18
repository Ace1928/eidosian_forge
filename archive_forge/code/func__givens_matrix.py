import numpy as np
import pennylane as qml
def _givens_matrix(a, b, left=True, tol=1e-08):
    """Build a :math:`2 \\times 2` Givens rotation matrix :math:`G`.

    When the matrix :math:`G` is applied to a vector :math:`[a,\\ b]^T` the following would happen:

    .. math::

            G \\times \\begin{bmatrix} a \\\\ b \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ r \\end{bmatrix} \\quad \\quad \\quad \\begin{bmatrix} a \\\\ b \\end{bmatrix} \\times G = \\begin{bmatrix} r \\\\ 0 \\end{bmatrix},

    where :math:`r` is a complex number.

    Args:
        a (float or complex): first element of the vector for which the Givens matrix is being computed
        b (float or complex): second element of the vector for which the Givens matrix is being computed
        left (bool): determines if the Givens matrix is being applied from the left side or right side.
        tol (float): determines tolerance limits for :math:`|a|` and :math:`|b|` under which they are considered as zero.

    Returns:
        np.ndarray (or tensor): Givens rotation matrix

    """
    abs_a, abs_b = (np.abs(a), np.abs(b))
    if abs_a < tol:
        cosine, sine, phase = (1.0, 0.0, 1.0)
    elif abs_b < tol:
        cosine, sine, phase = (0.0, 1.0, 1.0)
    else:
        hypot = np.hypot(abs_a, abs_b)
        cosine = abs_b / hypot
        sine = abs_a / hypot
        phase = 1.0 * b / abs_b * a.conjugate() / abs_a
    if left:
        return np.array([[phase * cosine, -sine], [phase * sine, cosine]])
    return np.array([[phase * sine, cosine], [-phase * cosine, sine]])