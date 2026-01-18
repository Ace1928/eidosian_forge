from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas
class BoolNot(object):

    def __init__(self, t):
        self.arg = t[0][1]

    def __bool__(self):
        v = bool(self.arg)
        return not v

    def __str__(self):
        return '~' + str(self.arg)
    __repr__ = __str__
    __nonzero__ = __bool__