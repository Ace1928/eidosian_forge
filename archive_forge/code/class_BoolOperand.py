from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas
class BoolOperand(object):

    def __init__(self, t):
        self.label = t[0]
        self.value = eval(t[0])

    def __bool__(self):
        return self.value

    def __str__(self):
        return self.label
    __repr__ = __str__
    __nonzero__ = __bool__