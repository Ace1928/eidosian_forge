class _TokenType(tuple):
    parent = None

    def split(self):
        buf = []
        node = self
        while node is not None:
            buf.append(node)
            node = node.parent
        buf.reverse()
        return buf

    def __init__(self, *args):
        self.subtypes = set()

    def __contains__(self, val):
        return self is val or (type(val) is self.__class__ and val[:len(self)] == self)

    def __getattr__(self, val):
        if not val or not val[0].isupper():
            return tuple.__getattribute__(self, val)
        new = _TokenType(self + (val,))
        setattr(self, val, new)
        self.subtypes.add(new)
        new.parent = self
        return new

    def __repr__(self):
        return 'Token' + (self and '.' or '') + '.'.join(self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self