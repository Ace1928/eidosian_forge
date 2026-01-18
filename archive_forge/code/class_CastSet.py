from collections import defaultdict
from functools import total_ordering
import enum
class CastSet(object):
    """A set of casting rules.

    There is at most one rule per target type.
    """

    def __init__(self):
        self._rels = {}

    def insert(self, to, rel):
        old = self.get(to)
        setrel = min(rel, old)
        self._rels[to] = setrel
        return old != setrel

    def items(self):
        return self._rels.items()

    def get(self, item):
        return self._rels.get(item, Conversion.nil)

    def __len__(self):
        return len(self._rels)

    def __repr__(self):
        body = ['{rel}({ty})'.format(rel=rel, ty=ty) for ty, rel in self._rels.items()]
        return '{' + ', '.join(body) + '}'

    def __contains__(self, item):
        return item in self._rels

    def __iter__(self):
        return iter(self._rels.keys())

    def __getitem__(self, item):
        return self._rels[item]