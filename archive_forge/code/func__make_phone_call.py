import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
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