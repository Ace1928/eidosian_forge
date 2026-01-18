from antlr3.constants import INVALID_TOKEN_TYPE
class MismatchedNotSetException(MismatchedSetException):
    """@brief Used for remote debugger deserialization"""

    def __str__(self):
        return 'MismatchedNotSetException(%r!=%r)' % (self.getUnexpectedType(), self.expecting)
    __repr__ = __str__