from antlr3.constants import INVALID_TOKEN_TYPE
class MismatchedTreeNodeException(RecognitionException):
    """@brief The next tree mode does not match the expected type."""

    def __init__(self, expecting, input):
        RecognitionException.__init__(self, input)
        self.expecting = expecting

    def __str__(self):
        return 'MismatchedTreeNodeException(%r!=%r)' % (self.getUnexpectedType(), self.expecting)
    __repr__ = __str__