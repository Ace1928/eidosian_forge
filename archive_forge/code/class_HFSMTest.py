import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
class HFSMTest(FSMTest):

    @staticmethod
    def _create_fsm(start_state, add_start=True, hierarchical=False, add_states=None):
        if hierarchical:
            m = machines.HierarchicalFiniteMachine()
        else:
            m = machines.FiniteMachine()
        if add_start:
            m.add_state(start_state)
            m.default_start_state = start_state
        if add_states:
            for s in add_states:
                if s not in m:
                    m.add_state(s)
        return m

    def _make_phone_call(self, talk_time=1.0):

        def phone_reaction(old_state, new_state, event, chat_iter):
            try:
                next(chat_iter)
            except StopIteration:
                return 'finish'
            else:
                return 'chat'
        talker = self._create_fsm('talk')
        talker.add_transition('talk', 'talk', 'pickup')
        talker.add_transition('talk', 'talk', 'chat')
        talker.add_reaction('talk', 'pickup', lambda *args: 'chat')
        chat_iter = iter(list(range(0, 10)))
        talker.add_reaction('talk', 'chat', phone_reaction, chat_iter)
        handler = self._create_fsm('begin', hierarchical=True)
        handler.add_state('phone', machine=talker)
        handler.add_state('hangup', terminal=True)
        handler.add_transition('begin', 'phone', 'call')
        handler.add_reaction('phone', 'call', lambda *args: 'pickup')
        handler.add_transition('phone', 'hangup', 'finish')
        return handler

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

    def test_nested_machines(self):
        dialer, _number_calling = self._make_phone_dialer()
        self.assertEqual(1, len(dialer.nested_machines))

    def test_nested_machine_initializers(self):
        dialer, _number_calling = self._make_phone_dialer()
        queried_for = []

        def init_with(nested_machine):
            queried_for.append(nested_machine)
            return None
        dialer.initialize(nested_start_state_fetcher=init_with)
        self.assertEqual(1, len(queried_for))

    def test_phone_dialer_iter(self):
        dialer, number_calling = self._make_phone_dialer()
        self.assertEqual(0, len(number_calling))
        r = runners.HierarchicalRunner(dialer)
        transitions = list(r.run_iter('dial'))
        self.assertEqual(('talk', 'hangup'), transitions[-1])
        self.assertEqual(len(number_calling), sum((1 if new_state == 'accumulate' else 0 for old_state, new_state in transitions)))
        self.assertEqual(10, len(number_calling))

    def test_phone_call(self):
        handler = self._make_phone_call()
        r = runners.HierarchicalRunner(handler)
        r.run('call')
        self.assertTrue(handler.terminated)

    def test_phone_call_iter(self):
        handler = self._make_phone_call()
        r = runners.HierarchicalRunner(handler)
        transitions = list(r.run_iter('call'))
        self.assertEqual(('talk', 'hangup'), transitions[-1])
        self.assertEqual(('begin', 'phone'), transitions[0])
        talk_talk = 0
        for transition in transitions:
            if transition == ('talk', 'talk'):
                talk_talk += 1
        self.assertGreater(talk_talk, 0)