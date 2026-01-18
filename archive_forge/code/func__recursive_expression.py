import pennylane as qml
from pennylane.operation import Operation
from pennylane.ops import Sum
from pennylane.ops.op_math import SProd
@qml.QueuingManager.stop_recording()
def _recursive_expression(x, order, ops):
    """Generate a list of operations using the
    recursive expression which defines the Trotter product.

    Args:
        x (complex): the evolution 'time'
        order (int): the order of the Trotter expansion
        ops (Iterable(~.Operators)): a list of terms in the Hamiltonian

    Returns:
        list: the approximation as product of exponentials of the Hamiltonian terms
    """
    if order == 1:
        return [qml.exp(op, x * 1j) for op in ops]
    if order == 2:
        return [qml.exp(op, x * 0.5j) for op in ops + ops[::-1]]
    scalar_1 = _scalar(order)
    scalar_2 = 1 - 4 * scalar_1
    ops_lst_1 = _recursive_expression(scalar_1 * x, order - 2, ops)
    ops_lst_2 = _recursive_expression(scalar_2 * x, order - 2, ops)
    return 2 * ops_lst_1 + ops_lst_2 + 2 * ops_lst_1