from antlr3.constants import INVALID_TOKEN_TYPE
class MissingTokenException(MismatchedTokenException):
    """
    We were expecting a token but it's not found.  The current token
    is actually what we wanted next.
    """

    def __init__(self, expecting, input, inserted):
        MismatchedTokenException.__init__(self, expecting, input)
        self.inserted = inserted

    def getMissingType(self):
        return self.expecting

    def __str__(self):
        if self.inserted is not None and self.token is not None:
            return 'MissingTokenException(inserted %r at %r)' % (self.inserted, self.token.text)
        if self.token is not None:
            return 'MissingTokenException(at %r)' % self.token.text
        return 'MissingTokenException'
    __repr__ = __str__