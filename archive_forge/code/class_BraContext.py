from antlr4 import *
from io import StringIO
import sys
class BraContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def L_ANGLE(self):
        return self.getToken(LaTeXParser.L_ANGLE, 0)

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def R_BAR(self):
        return self.getToken(LaTeXParser.R_BAR, 0)

    def BAR(self):
        return self.getToken(LaTeXParser.BAR, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_bra