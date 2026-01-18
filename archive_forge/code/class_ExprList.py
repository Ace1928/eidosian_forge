import sys
class ExprList(Node):
    __slots__ = ('exprs', 'coord', '__weakref__')

    def __init__(self, exprs, coord=None):
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(('exprs[%d]' % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in self.exprs or []:
            yield child
    attr_names = ()