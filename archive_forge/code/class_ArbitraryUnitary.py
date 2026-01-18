import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot
class ArbitraryUnitary(Operation):
    """Implements an arbitrary unitary on the specified wires.

    An arbitrary unitary on :math:`n` wires is parametrized by :math:`4^n - 1`
    independent real parameters. This templates uses Pauli word rotations to
    parametrize the unitary.

    **Example**

    ArbitraryUnitary can be used as a building block, e.g. to parametrize arbitrary
    two-qubit operations in a circuit:

    .. code-block:: python

        def arbitrary_nearest_neighbour_interaction(weights, wires):
            qml.broadcast(unitary=ArbitraryUnitary, pattern="double", wires=wires, parameters=weights)

    Args:
        weights (tensor_like): The angles of the Pauli word rotations, needs to have length :math:`4^n - 1`
            where :math:`n` is the number of wires the template acts upon.
        wires (Iterable): wires that the template acts on
    """
    num_wires = AnyWires
    grad_method = None
    num_params = 1
    ndim_params = (1,)

    def __init__(self, weights, wires, id=None):
        shape = qml.math.shape(weights)
        dim = 4 ** len(wires) - 1
        if len(shape) not in (1, 2) or shape[-1] != dim:
            raise ValueError(f'Weights tensor must be of shape {(dim,)} or (batch_dim, {dim}); got {shape}.')
        super().__init__(weights, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(weights, wires):
        """Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \\dots O_n.



        .. seealso:: :meth:`~.ArbitraryUnitary.decomposition`.

        Args:
            weights (tensor_like): The angles of the Pauli word rotations, needs to have length :math:`4^n - 1`
                    where :math:`n` is the number of wires the template acts upon.
            wires (Any or Iterable[Any]): wires that the operator acts on


        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []
        for i, pauli_word in enumerate(_all_pauli_words_but_identity(len(wires))):
            op_list.append(PauliRot(weights[..., i], pauli_word, wires=wires))
        return op_list

    @staticmethod
    def shape(n_wires):
        """Compute the expected shape of the weights tensor.

        Args:
            n_wires (int): number of wires that template acts on
        """
        return (4 ** n_wires - 1,)