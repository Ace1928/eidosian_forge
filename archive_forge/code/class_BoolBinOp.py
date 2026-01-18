from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas
class BoolBinOp(object):

    def __init__(self, t):
        self.args = t[0][0::2]

    def __str__(self):
        sep = ' %s ' % self.reprsymbol
        return '(' + sep.join(map(str, self.args)) + ')'

    def __bool__(self):
        return self.evalop((bool(a) for a in self.args))
    __nonzero__ = __bool__
    __repr__ = __str__