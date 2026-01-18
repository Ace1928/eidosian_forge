from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
class Mechanism(object):
    m = MethodicalMachine()

    def __init__(self):
        self.value = 1
        self.ranOutput = False

    @m.state(serialized='first-state', initial=True)
    def first(self):
        """First state."""

    @m.state(serialized='second-state')
    def second(self):
        """Second state."""

    @m.input()
    def input(self):
        """an input"""

    @m.output()
    def output(self):
        self.value = 2
        self.ranOutput = True
        return 1

    @m.output()
    def output2(self):
        return 2
    first.upon(input, second, [output], collector=lambda x: list(x)[0])
    second.upon(input, second, [output2], collector=lambda x: list(x)[0])

    @m.serializer()
    def save(self, state):
        return {'machine-state': state, 'some-value': self.value}

    @m.unserializer()
    def _restore(self, blob):
        self.value = blob['some-value']
        return blob['machine-state']

    @classmethod
    def fromBlob(cls, blob):
        self = cls()
        self._restore(blob)
        return self