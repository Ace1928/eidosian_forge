import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
class PauliError(Channel):
    """
    Pauli operator error channel for an arbitrary number of qubits.

    This channel is modelled by the following Kraus matrices:

    .. math::
        K_0 = \\sqrt{1-p} * I

    .. math::
        K_1 = \\sqrt{p} * (K_{w0} \\otimes K_{w1} \\otimes \\dots K_{wn})

    Where :math:`I` is the Identity,
    and :math:`\\otimes` denotes the Kronecker Product,
    and :math:`K_{wi}` denotes the Kraus matrix corresponding to the operator acting on wire :math:`wi`,
    and :math:`p` denotes the probability with which the channel is applied.

    .. warning::

        The size of the Kraus matrices for PauliError scale exponentially
        with the number of wires, the channel acts on. Simulations with
        PauliError can result in a significant increase in memory and
        computational usage. Use with caution!

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 3

    Args:
        operators (str): The Pauli operators acting on the specified (groups of) wires
        p (float): The probability of the operator being applied
        wires (Sequence[int] or int): The wires the channel acts on
        id (str or None): String representing the operation (optional)

    **Example:**

    >>> pe = PauliError("X", 0.5, wires=0)
    >>> km = pe.kraus_matrices()
    >>> km[0]
    array([[0.70710678, 0.        ],
           [0.        , 0.70710678]])
    >>> km[1]
        array([[0.        , 0.70710678],
               [0.70710678, 0.        ]])
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    num_params = 2
    'int: Number of trainable parameters that the operator depends on.'

    def __init__(self, operators, p, wires=None, id=None):
        super().__init__(operators, p, wires=wires, id=id)
        if not set(operators).issubset({'X', 'Y', 'Z'}):
            raise ValueError("The specified operators need to be either of 'X', 'Y' or 'Z'")
        if not np.is_abstract(p) and (not 0.0 <= p <= 1.0):
            raise ValueError('p must be in the interval [0,1]')
        if len(self.wires) != len(operators):
            raise ValueError('The number of operators must match the number of wires')
        nq = len(self.wires)
        if nq > 20:
            warnings.warn(f'The resulting Kronecker matrices will have dimensions {2 ** nq} x {2 ** nq}.\nThis equals {2 ** nq * 2 ** nq * 8 / 1024 ** 3} GB of physical memory for each matrix.')

    @staticmethod
    def compute_kraus_matrices(operators, p):
        """Kraus matrices representing the PauliError channel.

        Args:
            operators (str): the Pauli operators acting on the specified (groups of) wires
            p (float): probability of the operator being applied

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.PauliError.compute_kraus_matrices("X", 0.5)
        [array([[0.70710678, 0.        ], [0.        , 0.70710678]]),
         array([[0.        , 0.70710678], [0.70710678, 0.        ]])]
        """
        nq = len(operators)
        K0 = np.sqrt(1 - p + np.eps) * np.convert_like(np.cast_like(np.eye(2 ** nq), p), p)
        interface = np.get_interface(p)
        if interface == 'tensorflow' or 'Y' in operators:
            if interface == 'numpy':
                p = (1 + 0j) * p
            else:
                p = np.cast_like(p, 1j)
        ops = {'X': np.convert_like(np.cast_like(np.array([[0, 1], [1, 0]]), p), p), 'Y': np.convert_like(np.cast_like(np.array([[0, -1j], [1j, 0]]), p), p), 'Z': np.convert_like(np.cast_like(np.array([[1, 0], [0, -1]]), p), p)}
        K1 = np.sqrt(p + np.eps) * np.convert_like(np.cast_like(np.eye(1), p), p)
        for op in operators[::-1]:
            K1 = np.multi_dispatch()(np.kron)(ops[op], K1)
        return [K0, K1]