from antlr4 import *
from io import StringIO
import sys
class Abs_groupContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def BAR(self, i: int=None):
        if i is None:
            return self.getTokens(LaTeXParser.BAR)
        else:
            return self.getToken(LaTeXParser.BAR, i)

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_abs_group