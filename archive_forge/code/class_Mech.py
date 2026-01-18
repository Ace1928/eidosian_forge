from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
class Mech(object):
    m = MethodicalMachine()

    @m.input()
    def declaredInputName(self):
        """an input"""

    @m.state(initial=True)
    def aState(self):
        """state"""