class NamedType(Type):
    __slots__ = ('loc', 'name')
    _fields = ('name',)

    def __init__(self, name, loc=None):
        self.loc = loc
        self.name = name

    def __eq__(self, other):
        return self is other or (isinstance(other, NamedType) and self.name == other.name)

    def __repr__(self):
        return 'NamedType(name={self.name!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.loc)

    def __hash__(self):
        return id(self)