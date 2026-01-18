from antlr4 import *
from io import StringIO
import sys
class VarDecl2Context(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ID(self):
        return self.getToken(AutolevParser.ID, 0)

    def INT(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.INT)
        else:
            return self.getToken(AutolevParser.INT, i)

    def expr(self):
        return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_varDecl2

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterVarDecl2'):
            listener.enterVarDecl2(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitVarDecl2'):
            listener.exitVarDecl2(self)