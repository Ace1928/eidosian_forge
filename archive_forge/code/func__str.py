from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def _str(self, nodes):
    if self in nodes:
        return '(#{})'.format(nodes[self])
    else:
        nodes[self] = len(nodes)
        return '{} -> ({})'.format(self.name(), ', '.join((u._str(nodes.copy()) for u in self._users)))