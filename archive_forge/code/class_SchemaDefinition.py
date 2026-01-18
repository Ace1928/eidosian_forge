class SchemaDefinition(TypeSystemDefinition):
    __slots__ = ('loc', 'directives', 'operation_types')
    _fields = ('operation_types',)

    def __init__(self, operation_types, loc=None, directives=None):
        self.operation_types = operation_types
        self.loc = loc
        self.directives = directives

    def __eq__(self, other):
        return self is other or (isinstance(other, SchemaDefinition) and self.operation_types == other.operation_types and (self.directives == other.directives))

    def __repr__(self):
        return 'SchemaDefinition(operation_types={self.operation_types!r}, directives={self.directives!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.operation_types, self.loc, self.directives)

    def __hash__(self):
        return id(self)