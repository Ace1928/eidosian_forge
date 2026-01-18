from antlr4 import *
from io import StringIO
import sys
class BinomContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.n = None
        self.k = None

    def L_BRACE(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.L_BRACE)
        else:
            return self.getToken(LaTeXParser.L_BRACE, i)

    def R_BRACE(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.R_BRACE)
        else:
            return self.getToken(LaTeXParser.R_BRACE, i)

    def CMD_BINOM(self):
        return self.getToken(LaTeXParser.CMD_BINOM, 0)

    def CMD_DBINOM(self):
        return self.getToken(LaTeXParser.CMD_DBINOM, 0)

    def CMD_TBINOM(self):
        return self.getToken(LaTeXParser.CMD_TBINOM, 0)

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.ExprContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.ExprContext, i)

    def getRuleIndex(self):
        return LaTeXParser.RULE_binom