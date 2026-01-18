class UnionTypeDefinition(TypeDefinition):
    __slots__ = ('loc', 'name', 'types', 'directives')
    _fields = ('name', 'types')

    def __init__(self, name, types, loc=None, directives=None):
        self.loc = loc
        self.name = name
        self.types = types
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, UnionTypeDefinition) and self.name == other.name and (self.types == other.types) and (self.directives == other.directives))

    def __repr__(self):
        return 'UnionTypeDefinition(name={self.name!r}, types={self.types!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.types, self.loc, self.directives)

    def __hash__(self):
        return id(self)