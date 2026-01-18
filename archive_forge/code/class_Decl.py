import sys
class Decl(Node):
    __slots__ = ('name', 'quals', 'align', 'storage', 'funcspec', 'type', 'init', 'bitsize', 'coord', '__weakref__')

    def __init__(self, name, quals, align, storage, funcspec, type, init, bitsize, coord=None):
        self.name = name
        self.quals = quals
        self.align = align
        self.storage = storage
        self.funcspec = funcspec
        self.type = type
        self.init = init
        self.bitsize = bitsize
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(('type', self.type))
        if self.init is not None:
            nodelist.append(('init', self.init))
        if self.bitsize is not None:
            nodelist.append(('bitsize', self.bitsize))
        return tuple(nodelist)

    def __iter__(self):
        if self.type is not None:
            yield self.type
        if self.init is not None:
            yield self.init
        if self.bitsize is not None:
            yield self.bitsize
    attr_names = ('name', 'quals', 'align', 'storage', 'funcspec')