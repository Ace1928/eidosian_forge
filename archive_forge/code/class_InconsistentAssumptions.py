from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
class InconsistentAssumptions(ValueError):

    def __str__(self):
        kb, fact, value = self.args
        return '%s, %s=%s' % (kb, fact, value)