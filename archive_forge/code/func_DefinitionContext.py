from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
@contextmanager
def DefinitionContext(self, node):
    self._currenthead.append(node)
    self._definitions.append(defaultdict(ordered_set))
    self._promoted_locals.append(set())
    yield
    self._promoted_locals.pop()
    self._definitions.pop()
    self._currenthead.pop()