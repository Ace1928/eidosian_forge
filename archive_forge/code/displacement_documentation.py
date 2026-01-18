import pennylane as qml
from pennylane.operation import Operation, AnyWires
Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.DisplacementEmbedding.decomposition`.

        Args:
            pars (tensor_like): parameters extracted from features and constant
            wires (Any or Iterable[Any]): wires that the template acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> pars = torch.tensor([[1., 0.], [2., 0.]])
        >>> qml.DisplacementEmbedding.compute_decomposition(pars, wires=[0, 1])
        [Displacement(tensor(1.), tensor(0.), wires=[0]),
         Displacement(tensor(2.), tensor(0.), wires=[1])]
        