from antlr4 import *
from io import StringIO
import sys
class MatrixContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.ExprContext)
        else:
            return self.getTypedRuleContext(AutolevParser.ExprContext, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_matrix

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterMatrix'):
            listener.enterMatrix(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitMatrix'):
            listener.exitMatrix(self)