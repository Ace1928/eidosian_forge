import sys
class ArrayDecl(Node):
    __slots__ = ('type', 'dim', 'dim_quals', 'coord', '__weakref__')

    def __init__(self, type, dim, dim_quals, coord=None):
        self.type = type
        self.dim = dim
        self.dim_quals = dim_quals
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(('type', self.type))
        if self.dim is not None:
            nodelist.append(('dim', self.dim))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.dim is not None:
            yield self.dim
    attr_names = ('dim_quals',)