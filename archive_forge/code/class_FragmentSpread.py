class FragmentSpread(Selection):
    __slots__ = ('loc', 'name', 'directives')
    _fields = ('name', 'directives')

    def __init__(self, name, directives=None, loc=None):
        self.loc = loc
        self.name = name
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, FragmentSpread) and self.name == other.name and (self.directives == other.directives))

    def __repr__(self):
        return 'FragmentSpread(name={self.name!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.directives, self.loc)

    def __hash__(self):
        return id(self)