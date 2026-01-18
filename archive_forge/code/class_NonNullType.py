class NonNullType(Type):
    __slots__ = ('loc', 'type')
    _fields = ('type',)

    def __init__(self, type, loc=None):
        self.loc = loc
        self.type = type

    def __eq__(self, other):
        return self is other or (isinstance(other, NonNullType) and self.type == other.type)

    def __repr__(self):
        return 'NonNullType(type={self.type!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.type, self.loc)

    def __hash__(self):
        return id(self)