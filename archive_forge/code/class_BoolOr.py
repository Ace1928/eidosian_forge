from pyparsing import infixNotation, opAssoc, Keyword, Word, alphas
class BoolOr(BoolBinOp):
    reprsymbol = '|'
    evalop = any