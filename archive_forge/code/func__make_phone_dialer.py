import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def _make_phone_dialer(self):
    dialer = self._create_fsm('idle', hierarchical=True)
    digits = self._create_fsm('idle')
    dialer.add_state('pickup', machine=digits)
    dialer.add_transition('idle', 'pickup', 'dial')
    dialer.add_reaction('pickup', 'dial', lambda *args: 'press')
    dialer.add_state('hangup', terminal=True)

    def react_to_press(last_state, new_state, event, number_calling):
        if len(number_calling) >= 10:
            return 'call'
        else:
            return 'press'
    digit_maker = functools.partial(random.randint, 0, 9)
    number_calling = []
    digits.add_state('accumulate', on_enter=lambda *args: number_calling.append(digit_maker()))
    digits.add_transition('idle', 'accumulate', 'press')
    digits.add_transition('accumulate', 'accumulate', 'press')
    digits.add_reaction('accumulate', 'press', react_to_press, number_calling)
    digits.add_state('dial', terminal=True)
    digits.add_transition('accumulate', 'dial', 'call')
    digits.add_reaction('dial', 'call', lambda *args: 'ringing')
    dialer.add_state('talk')
    dialer.add_transition('pickup', 'talk', 'ringing')
    dialer.add_reaction('talk', 'ringing', lambda *args: 'hangup')
    dialer.add_transition('talk', 'hangup', 'hangup')
    return (dialer, number_calling)