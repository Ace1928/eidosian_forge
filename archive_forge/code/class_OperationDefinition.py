class OperationDefinition(Definition):
    __slots__ = ('loc', 'operation', 'name', 'variable_definitions', 'directives', 'selection_set')
    _fields = ('operation', 'name', 'variable_definitions', 'directives', 'selection_set')

    def __init__(self, operation, selection_set, name=None, variable_definitions=None, directives=None, loc=None):
        self.loc = loc
        self.operation = operation
        self.name = name
        self.variable_definitions = variable_definitions
        self.directives = directives
        self.selection_set = selection_set

    def __eq__(self, other):
        return self is other or (isinstance(other, OperationDefinition) and self.operation == other.operation and (self.name == other.name) and (self.variable_definitions == other.variable_definitions) and (self.directives == other.directives) and (self.selection_set == other.selection_set))

    def __repr__(self):
        return 'OperationDefinition(operation={self.operation!r}, name={self.name!r}, variable_definitions={self.variable_definitions!r}, directives={self.directives!r}, selection_set={self.selection_set!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.operation, self.selection_set, self.name, self.variable_definitions, self.directives, self.loc)

    def __hash__(self):
        return id(self)