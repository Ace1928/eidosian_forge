from antlr3.constants import INVALID_TOKEN_TYPE
class UnwantedTokenException(MismatchedTokenException):
    """An extra token while parsing a TokenStream"""

    def getUnexpectedToken(self):
        return self.token

    def __str__(self):
        exp = ', expected %s' % self.expecting
        if self.expecting == INVALID_TOKEN_TYPE:
            exp = ''
        if self.token is None:
            return 'UnwantedTokenException(found=%s%s)' % (None, exp)
        return 'UnwantedTokenException(found=%s%s)' % (self.token.text, exp)
    __repr__ = __str__