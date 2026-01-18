import sys
class Alignas(Node):
    __slots__ = ('alignment', 'coord', '__weakref__')

    def __init__(self, alignment, coord=None):
        self.alignment = alignment
        self.coord = coord

    def children(self):
        nodelist = []
        if self.alignment is not None:
            nodelist.append(('alignment', self.alignment))
        return tuple(nodelist)

    def __iter__(self):
        if self.alignment is not None:
            yield self.alignment
    attr_names = ()