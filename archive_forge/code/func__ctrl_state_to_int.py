import numpy
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from .parametervector import ParameterVectorElement
def _ctrl_state_to_int(ctrl_state, num_ctrl_qubits):
    """Convert ctrl_state to int.

    Args:
        ctrl_state (None, str, int): ctrl_state. If None, set to 2**num_ctrl_qubits-1.
            If str, convert to int. If int, pass.
        num_ctrl_qubits (int): The number of control qubits.

    Return:
        int: ctrl_state

    Raises:
        CircuitError: invalid ctrl_state
    """
    ctrl_state_std = None
    if isinstance(ctrl_state, str):
        try:
            assert len(ctrl_state) == num_ctrl_qubits
            ctrl_state = int(ctrl_state, 2)
        except ValueError as ex:
            raise CircuitError('invalid control bit string: ' + ctrl_state) from ex
        except AssertionError as ex:
            raise CircuitError('invalid control bit string: length != num_ctrl_qubits') from ex
    if isinstance(ctrl_state, int):
        if 0 <= ctrl_state < 2 ** num_ctrl_qubits:
            ctrl_state_std = ctrl_state
        else:
            raise CircuitError('invalid control state specification')
    elif ctrl_state is None:
        ctrl_state_std = 2 ** num_ctrl_qubits - 1
    else:
        raise CircuitError(f'invalid control state specification: {repr(ctrl_state)}')
    return ctrl_state_std