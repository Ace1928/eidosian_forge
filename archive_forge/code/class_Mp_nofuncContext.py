from antlr4 import *
from io import StringIO
import sys
class Mp_nofuncContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def unary_nofunc(self):
        return self.getTypedRuleContext(LaTeXParser.Unary_nofuncContext, 0)

    def mp_nofunc(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.Mp_nofuncContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.Mp_nofuncContext, i)

    def MUL(self):
        return self.getToken(LaTeXParser.MUL, 0)

    def CMD_TIMES(self):
        return self.getToken(LaTeXParser.CMD_TIMES, 0)

    def CMD_CDOT(self):
        return self.getToken(LaTeXParser.CMD_CDOT, 0)

    def DIV(self):
        return self.getToken(LaTeXParser.DIV, 0)

    def CMD_DIV(self):
        return self.getToken(LaTeXParser.CMD_DIV, 0)

    def COLON(self):
        return self.getToken(LaTeXParser.COLON, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_mp_nofunc