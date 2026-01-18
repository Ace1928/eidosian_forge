import sys
class EnumeratorList(Node):
    __slots__ = ('enumerators', 'coord', '__weakref__')

    def __init__(self, enumerators, coord=None):
        self.enumerators = enumerators
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.enumerators or []):
            nodelist.append(('enumerators[%d]' % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in self.enumerators or []:
            yield child
    attr_names = ()