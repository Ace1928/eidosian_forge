from antlr4 import *
from io import StringIO
import sys
class MassDeclContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def Mass(self):
        return self.getToken(AutolevParser.Mass, 0)

    def massDecl2(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.MassDecl2Context)
        else:
            return self.getTypedRuleContext(AutolevParser.MassDecl2Context, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_massDecl

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterMassDecl'):
            listener.enterMassDecl(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitMassDecl'):
            listener.exitMassDecl(self)