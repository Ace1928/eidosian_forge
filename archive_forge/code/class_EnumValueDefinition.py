class EnumValueDefinition(Node):
    __slots__ = ('loc', 'name', 'directives')
    _fields = ('name',)

    def __init__(self, name, loc=None, directives=None):
        self.loc = loc
        self.name = name
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, EnumValueDefinition) and self.name == other.name and (self.directives == other.directives))

    def __repr__(self):
        return 'EnumValueDefinition(name={self.name!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.loc, self.directives)

    def __hash__(self):
        return id(self)