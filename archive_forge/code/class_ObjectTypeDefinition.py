class ObjectTypeDefinition(TypeDefinition):
    __slots__ = ('loc', 'name', 'interfaces', 'directives', 'fields')
    _fields = ('name', 'interfaces', 'fields')

    def __init__(self, name, fields, interfaces=None, loc=None, directives=None):
        self.loc = loc
        self.name = name
        self.interfaces = interfaces
        self.fields = fields
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, ObjectTypeDefinition) and self.name == other.name and (self.interfaces == other.interfaces) and (self.fields == other.fields) and (self.directives == other.directives))

    def __repr__(self):
        return 'ObjectTypeDefinition(name={self.name!r}, interfaces={self.interfaces!r}, fields={self.fields!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.name, self.fields, self.interfaces, self.loc, self.directives)

    def __hash__(self):
        return id(self)