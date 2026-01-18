import sys
class EllipsisParam(Node):
    __slots__ = ('coord', '__weakref__')

    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()

    def __iter__(self):
        return
        yield
    attr_names = ()