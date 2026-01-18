import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
class PhaseFlip(Channel):
    """
    Single-qubit bit flip (Pauli :math:`Z`) error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \\sqrt{1-p} \\begin{bmatrix}
                1 & 0 \\\\
                0 & 1
                \\end{bmatrix}

    .. math::
        K_1 = \\sqrt{p}\\begin{bmatrix}
                1 & 0  \\\\
                0 & -1
                \\end{bmatrix}

    where :math:`p \\in [0, 1]` is the probability of a phase flip (Pauli :math:`Z`) error.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): The probability that a phase flip error occurs.
        wires (Sequence[int] or int): the wire the channel acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = 'A'
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):
        """Kraus matrices representing the PhaseFlip channel.

        Args:
            p (float): the probability that a phase flip error occurs

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.PhaseFlip.compute_kraus_matrices(0.5)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[ 0.70710678,  0.        ], [ 0.        , -0.70710678]])]
        """
        if not np.is_abstract(p) and (not 0.0 <= p <= 1.0):
            raise ValueError('p must be in the interval [0,1]')
        K0 = np.sqrt(1 - p + np.eps) * np.convert_like(np.cast_like(np.eye(2), p), p)
        K1 = np.sqrt(p + np.eps) * np.convert_like(np.cast_like(np.diag([1, -1]), p), p)
        return [K0, K1]