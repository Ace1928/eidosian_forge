from ..grammar import NonTerminal, Terminal
class TransitiveItem(Item):
    __slots__ = ('recognized', 'reduction', 'column', 'next_titem')

    def __init__(self, recognized, trule, originator, start):
        super(TransitiveItem, self).__init__(trule.rule, trule.ptr, trule.start)
        self.recognized = recognized
        self.reduction = originator
        self.column = start
        self.next_titem = None
        self._hash = hash((self.s, self.start, self.recognized))

    def __eq__(self, other):
        if not isinstance(other, TransitiveItem):
            return False
        return self is other or (type(self.s) == type(other.s) and self.s == other.s and (self.start == other.start) and (self.recognized == other.recognized))

    def __hash__(self):
        return self._hash

    def __repr__(self):
        before = (expansion.name for expansion in self.rule.expansion[:self.ptr])
        after = (expansion.name for expansion in self.rule.expansion[self.ptr:])
        return '{} : {} -> {}* {} ({}, {})'.format(self.recognized.name, self.rule.origin.name, ' '.join(before), ' '.join(after), self.column, self.start)