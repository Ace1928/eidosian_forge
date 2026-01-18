from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
class MechanismWithDocstring(object):
    m = MethodicalMachine()

    @m.input()
    def input(self):
        """an input"""

    @m.state(initial=True)
    def start(self):
        """starting state"""
    start.upon(input, enter=start, outputs=[])