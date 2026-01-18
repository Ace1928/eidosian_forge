import sys
class StaticAssert(Node):
    __slots__ = ('cond', 'message', 'coord', '__weakref__')

    def __init__(self, cond, message, coord=None):
        self.cond = cond
        self.message = message
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None:
            nodelist.append(('cond', self.cond))
        if self.message is not None:
            nodelist.append(('message', self.message))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.message is not None:
            yield self.message
    attr_names = ()