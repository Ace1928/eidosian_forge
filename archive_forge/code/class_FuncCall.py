import sys
class FuncCall(Node):
    __slots__ = ('name', 'args', 'coord', '__weakref__')

    def __init__(self, name, args, coord=None):
        self.name = name
        self.args = args
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None:
            nodelist.append(('name', self.name))
        if self.args is not None:
            nodelist.append(('args', self.args))
        return tuple(nodelist)

    def __iter__(self):
        if self.name is not None:
            yield self.name
        if self.args is not None:
            yield self.args
    attr_names = ()