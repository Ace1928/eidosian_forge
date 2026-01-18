import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumScript
def _convert_op_to_numpy_data(op: qml.operation.Operator) -> qml.operation.Operator:
    if math.get_interface(*op.data) == 'numpy':
        return op
    return qml.ops.functions.bind_new_parameters(op, math.unwrap(op.data))