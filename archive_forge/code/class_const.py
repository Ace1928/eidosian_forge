from twisted.python import roots
from twisted.trial import unittest
class const(roots.Constrained):

    def nameConstraint(self, name: str) -> bool:
        return name == 'x'