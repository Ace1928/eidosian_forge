class OperationTypeDefinition(Node):
    __slots__ = ('loc', 'operation', 'type')
    _fields = ('operation', 'type')

    def __init__(self, operation, type, loc=None):
        self.operation = operation
        self.type = type
        self.loc = loc

    def __eq__(self, other):
        return self is other or (isinstance(other, OperationTypeDefinition) and self.operation == other.operation and (self.type == other.type))

    def __repr__(self):
        return 'OperationTypeDefinition(operation={self.operation!r}, type={self.type!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.operation, self.type, self.loc)

    def __hash__(self):
        return id(self)