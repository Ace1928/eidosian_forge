from antlr3.constants import INVALID_TOKEN_TYPE
class MismatchedSetException(RecognitionException):
    """@brief The next token does not match a set of expected types."""

    def __init__(self, expecting, input):
        RecognitionException.__init__(self, input)
        self.expecting = expecting

    def __str__(self):
        return 'MismatchedSetException(%r not in %r)' % (self.getUnexpectedType(), self.expecting)
    __repr__ = __str__