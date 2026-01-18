from collections import defaultdict
from functools import total_ordering
import enum
class TypeGraph(object):
    """A graph that maintains the casting relationship of all types.

    This simplifies the definition of casting rules by automatically
    propagating the rules.
    """

    def __init__(self, callback=None):
        """
        Args
        ----
        - callback: callable or None
            It is called for each new casting rule with
            (from_type, to_type, castrel).
        """
        assert callback is None or callable(callback)
        self._forwards = defaultdict(CastSet)
        self._backwards = defaultdict(set)
        self._callback = callback

    def get(self, ty):
        return self._forwards[ty]

    def propagate(self, a, b, baserel):
        backset = self._backwards[a]
        for child in self._forwards[b]:
            rel = max(baserel, self._forwards[b][child])
            if a != child:
                if self._forwards[a].insert(child, rel):
                    self._callback(a, child, rel)
                self._backwards[child].add(a)
            for backnode in backset:
                if backnode != child:
                    backrel = max(rel, self._forwards[backnode][a])
                    if self._forwards[backnode].insert(child, backrel):
                        self._callback(backnode, child, backrel)
                    self._backwards[child].add(backnode)
        for child in self._backwards[a]:
            rel = max(baserel, self._forwards[child][a])
            if b != child:
                if self._forwards[child].insert(b, rel):
                    self._callback(child, b, rel)
                self._backwards[b].add(child)

    def insert_rule(self, a, b, rel):
        self._forwards[a].insert(b, rel)
        self._callback(a, b, rel)
        self._backwards[b].add(a)
        self.propagate(a, b, rel)

    def promote(self, a, b):
        self.insert_rule(a, b, Conversion.promote)

    def safe(self, a, b):
        self.insert_rule(a, b, Conversion.safe)

    def unsafe(self, a, b):
        self.insert_rule(a, b, Conversion.unsafe)