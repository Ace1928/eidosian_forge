from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas
class BoolAnd(BoolBinOp):
    reprsymbol = '&'
    evalop = all