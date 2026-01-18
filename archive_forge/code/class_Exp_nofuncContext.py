from antlr4 import *
from io import StringIO
import sys
class Exp_nofuncContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def comp_nofunc(self):
        return self.getTypedRuleContext(LaTeXParser.Comp_nofuncContext, 0)

    def exp_nofunc(self):
        return self.getTypedRuleContext(LaTeXParser.Exp_nofuncContext, 0)

    def CARET(self):
        return self.getToken(LaTeXParser.CARET, 0)

    def atom(self):
        return self.getTypedRuleContext(LaTeXParser.AtomContext, 0)

    def L_BRACE(self):
        return self.getToken(LaTeXParser.L_BRACE, 0)

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def R_BRACE(self):
        return self.getToken(LaTeXParser.R_BRACE, 0)

    def subexpr(self):
        return self.getTypedRuleContext(LaTeXParser.SubexprContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_exp_nofunc