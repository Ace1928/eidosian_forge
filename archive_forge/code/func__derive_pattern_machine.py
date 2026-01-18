from dissononce.extras.processing.handshakestate_forwarder import ForwarderHandshakeState
from transitions import Machine
from transitions.core import MachineError
import logging
def _derive_pattern_machine(self, handshake_pattern, initiator):
    """
        :param pattern:
        :type pattern: HandshakePattern
        :return:
        :rtype: Machine
        """
    states = ['finish']
    transitions = []
    prev_state = None
    for i in range(0, len(handshake_pattern.message_patterns)):
        read_phase = i % 2 == 0
        if handshake_pattern.interpret_as_bob:
            read_phase = not read_phase
        if not initiator:
            read_phase = not read_phase
        message_pattern = handshake_pattern.message_patterns[i]
        pattern_str = '_'.join(message_pattern)
        template = self._TEMPLATE_PATTERN_STATE_WRITE if read_phase else self._TEMPLATE_PATTERN_STATE_READ
        state = template.format(pattern=pattern_str)
        if prev_state is not None:
            action = 'read' if read_phase else 'write'
            transitions.append([action, prev_state, state])
        if i == len(handshake_pattern.message_patterns) - 1:
            transitions.append(['write' if read_phase else 'read', state, 'finish'])
        states.append(state)
        prev_state = state
    return Machine(states=states, transitions=transitions, initial=states[1])