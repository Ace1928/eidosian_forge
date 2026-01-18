from antlr4 import *
from io import StringIO
import sys
class InertiaDeclContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def Inertia(self):
        return self.getToken(AutolevParser.Inertia, 0)

    def ID(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.ID)
        else:
            return self.getToken(AutolevParser.ID, i)

    def expr(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.ExprContext)
        else:
            return self.getTypedRuleContext(AutolevParser.ExprContext, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_inertiaDecl

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterInertiaDecl'):
            listener.enterInertiaDecl(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitInertiaDecl'):
            listener.exitInertiaDecl(self)