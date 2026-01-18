import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
class ResetError(Channel):
    """
    Single-qubit Reset error channel.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \\sqrt{1-p_0-p_1} \\begin{bmatrix}
                1 & 0 \\\\
                0 & 1
                \\end{bmatrix}

    .. math::
        K_1 = \\sqrt{p_0}\\begin{bmatrix}
                1 & 0  \\\\
                0 & 0
                \\end{bmatrix}

    .. math::
        K_2 = \\sqrt{p_0}\\begin{bmatrix}
                0 & 1  \\\\
                0 & 0
                \\end{bmatrix}

    .. math::
        K_3 = \\sqrt{p_1}\\begin{bmatrix}
                0 & 0  \\\\
                1 & 0
                \\end{bmatrix}

    .. math::
        K_4 = \\sqrt{p_1}\\begin{bmatrix}
                0 & 0  \\\\
                0 & 1
                \\end{bmatrix}

    where :math:`p_0 \\in [0, 1]` is the probability of a reset to 0,
    and :math:`p_1 \\in [0, 1]` is the probability of a reset to 1 error.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2

    Args:
        p_0 (float): The probability that a reset to 0 error occurs.
        p_1 (float): The probability that a reset to 1 error occurs.
        wires (Sequence[int] or int): the wire the channel acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 2
    num_wires = 1
    grad_method = 'F'

    def __init__(self, p0, p1, wires, id=None):
        super().__init__(p0, p1, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p_0, p_1):
        """Kraus matrices representing the ResetError channel.

        Args:
            p_0 (float): probability that a reset to 0 error occurs
            p_1 (float): probability that a reset to 1 error occurs

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.ResetError.compute_kraus_matrices(0.2, 0.3)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[0.4472136, 0.       ], [0.       , 0.       ]]),
         array([[0.       , 0.4472136], [0.       , 0.       ]]),
         array([[0.        , 0.        ], [0.54772256, 0.        ]]),
         array([[0.        , 0.        ], [0.        , 0.54772256]])]
        """
        if not np.is_abstract(p_0) and (not 0.0 <= p_0 <= 1.0):
            raise ValueError('p_0 must be in the interval [0,1]')
        if not np.is_abstract(p_1) and (not 0.0 <= p_1 <= 1.0):
            raise ValueError('p_1 must be in the interval [0,1]')
        if not np.is_abstract(p_0 + p_1) and (not 0.0 <= p_0 + p_1 <= 1.0):
            raise ValueError('p_0 + p_1 must be in the interval [0,1]')
        interface = np.get_interface(p_0, p_1)
        p_0, p_1 = np.coerce([p_0, p_1], like=interface)
        K0 = np.sqrt(1 - p_0 - p_1 + np.eps) * np.convert_like(np.cast_like(np.eye(2), p_0), p_0)
        K1 = np.sqrt(p_0 + np.eps) * np.convert_like(np.cast_like(np.array([[1, 0], [0, 0]]), p_0), p_0)
        K2 = np.sqrt(p_0 + np.eps) * np.convert_like(np.cast_like(np.array([[0, 1], [0, 0]]), p_0), p_0)
        K3 = np.sqrt(p_1 + np.eps) * np.convert_like(np.cast_like(np.array([[0, 0], [1, 0]]), p_0), p_0)
        K4 = np.sqrt(p_1 + np.eps) * np.convert_like(np.cast_like(np.array([[0, 0], [0, 1]]), p_0), p_0)
        return [K0, K1, K2, K3, K4]