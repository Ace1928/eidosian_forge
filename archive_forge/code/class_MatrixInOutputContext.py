from antlr4 import *
from io import StringIO
import sys
class MatrixInOutputContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ID(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.ID)
        else:
            return self.getToken(AutolevParser.ID, i)

    def FLOAT(self):
        return self.getToken(AutolevParser.FLOAT, 0)

    def INT(self):
        return self.getToken(AutolevParser.INT, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_matrixInOutput

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterMatrixInOutput'):
            listener.enterMatrixInOutput(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitMatrixInOutput'):
            listener.exitMatrixInOutput(self)