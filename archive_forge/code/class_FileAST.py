import sys
class FileAST(Node):
    __slots__ = ('ext', 'coord', '__weakref__')

    def __init__(self, ext, coord=None):
        self.ext = ext
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.ext or []):
            nodelist.append(('ext[%d]' % i, child))
        return tuple(nodelist)

    def __iter__(self):
        for child in self.ext or []:
            yield child
    attr_names = ()