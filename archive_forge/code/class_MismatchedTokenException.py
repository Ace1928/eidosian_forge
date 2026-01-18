from antlr3.constants import INVALID_TOKEN_TYPE
class MismatchedTokenException(RecognitionException):
    """@brief A mismatched char or Token or tree node."""

    def __init__(self, expecting, input):
        RecognitionException.__init__(self, input)
        self.expecting = expecting

    def __str__(self):
        return 'MismatchedTokenException(%r!=%r)' % (self.getUnexpectedType(), self.expecting)
    __repr__ = __str__