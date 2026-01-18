from antlr4 import *
from io import StringIO
import sys
class MassDecl2Context(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ID(self):
        return self.getToken(AutolevParser.ID, 0)

    def expr(self):
        return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_massDecl2

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterMassDecl2'):
            listener.enterMassDecl2(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitMassDecl2'):
            listener.exitMassDecl2(self)