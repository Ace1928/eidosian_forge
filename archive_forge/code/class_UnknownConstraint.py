class UnknownConstraint(Exception):
    __slots__ = ('constraint',)

    def __init__(self, constraint):
        self.constraint = constraint