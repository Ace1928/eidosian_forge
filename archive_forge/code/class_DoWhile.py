import sys
class DoWhile(Node):
    __slots__ = ('cond', 'stmt', 'coord', '__weakref__')

    def __init__(self, cond, stmt, coord=None):
        self.cond = cond
        self.stmt = stmt
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None:
            nodelist.append(('cond', self.cond))
        if self.stmt is not None:
            nodelist.append(('stmt', self.stmt))
        return tuple(nodelist)

    def __iter__(self):
        if self.cond is not None:
            yield self.cond
        if self.stmt is not None:
            yield self.stmt
    attr_names = ()