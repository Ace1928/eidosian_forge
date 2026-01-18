import sys
class PtrDecl(Node):
    __slots__ = ('quals', 'type', 'coord', '__weakref__')

    def __init__(self, quals, type, coord=None):
        self.quals = quals
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(('type', self.type))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
    attr_names = ('quals',)