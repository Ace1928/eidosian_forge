from antlr3.constants import INVALID_TOKEN_TYPE
class MismatchedRangeException(RecognitionException):
    """@brief The next token does not match a range of expected types."""

    def __init__(self, a, b, input):
        RecognitionException.__init__(self, input)
        self.a = a
        self.b = b

    def __str__(self):
        return 'MismatchedRangeException(%r not in [%r..%r])' % (self.getUnexpectedType(), self.a, self.b)
    __repr__ = __str__