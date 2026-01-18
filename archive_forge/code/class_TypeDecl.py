import sys
class TypeDecl(Node):
    __slots__ = ('declname', 'quals', 'align', 'type', 'coord', '__weakref__')

    def __init__(self, declname, quals, align, type, coord=None):
        self.declname = declname
        self.quals = quals
        self.align = align
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
    attr_names = ('declname', 'quals', 'align')