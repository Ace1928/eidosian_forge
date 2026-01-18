import sys
class NamedInitializer(Node):
    __slots__ = ('name', 'expr', 'coord', '__weakref__')

    def __init__(self, name, expr, coord=None):
        self.name = name
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(('expr', self.expr))
        for i, child in enumerate(self.name or []):
            nodelist.append(('name[%d]' % i, child))
        return tuple(nodelist)

    def __iter__(self):
        if self.expr is not None:
            yield self.expr
        for child in self.name or []:
            yield child
    attr_names = ()