import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
@classmethod
def _effect_builder(cls, new_state, event):
    return cls.Effect(new_state['reactions'].get(event), new_state['terminal'], new_state.get('machine'))