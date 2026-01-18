import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
class _Jump(object):
    """A FSM transition tracks this data while jumping."""

    def __init__(self, name, on_enter, on_exit):
        self.name = name
        self.on_enter = on_enter
        self.on_exit = on_exit