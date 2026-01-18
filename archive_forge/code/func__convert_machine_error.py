from dissononce.extras.processing.handshakestate_forwarder import ForwarderHandshakeState
from transitions import Machine
from transitions.core import MachineError
import logging
def _convert_machine_error(self, machine_error, bad_method):
    """
        :param machine_error:
        :type machine_error: MachineError
        :param bad_method:
        :type bad_method: str
        :return:
        :rtype:
        """
    if self._handshake_machine.state == 'init':
        current = 'initialize'
    elif self._handshake_machine.state == 'handshake':
        current = 'write_message' if bad_method == 'read_message' else 'write_message'
    else:
        current = self._handshake_machine.state
    error_message = self.ERROR_TEMPL.format(bad_method=bad_method, current=current)
    return AssertionError(error_message)