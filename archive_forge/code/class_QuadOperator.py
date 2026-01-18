import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class QuadOperator(CVObservable):
    """
    The generalized quadrature observable :math:`\\x_\\phi = \\x cos\\phi+\\p\\sin\\phi`.

    When used with the :func:`~pennylane.expval` function, the expectation
    value :math:`\\braket{\\hat{\\x_\\phi}}` is returned. This corresponds to
    the mean displacement in the phase space along axis at angle :math:`\\phi`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Observable order: 1st order in the quadrature operators
    * Heisenberg representation:

      .. math:: d = [0, \\cos\\phi, \\sin\\phi]

    Args:
        phi (float): axis in the phase space at which to calculate
            the generalized quadrature observable
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = 'A'
    ev_order = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        phi = p[0]
        return np.array([0, math.cos(phi), math.sin(phi)])

    def label(self, decimals=None, base_label=None, cache=None):
        """A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.QuadOperator(1.234, wires=0)
        >>> op.label()
        'cos(φ)x\\n+sin(φ)p'
        >>> op.label(decimals=2)
        'cos(1.23)x\\n+sin(1.23)p'
        >>> op.label(base_label="Quad", decimals=2)
        'Quad\\n(1.23)'

        """
        if base_label is not None:
            return super().label(decimals=decimals, base_label=base_label, cache=cache)
        if decimals is None:
            p = 'φ'
        else:
            p = format(qml_math.array(self.parameters[0]), f'.{decimals}f')
        return f'cos({p})x\n+sin({p})p'