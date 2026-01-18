from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
class Machination(object):
    machine = MethodicalMachine()
    counter = 0

    @machine.input()
    def anInput(self):
        """an input"""

    @machine.output()
    def anOutput(self):
        self.counter += 1

    @machine.state(initial=True)
    def state(self):
        """a machine state"""
    state.upon(anInput, enter=state, outputs=[anOutput])