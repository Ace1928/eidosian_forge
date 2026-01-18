from antlr4 import *
from io import StringIO
import sys
class VarDeclContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def varType(self):
        return self.getTypedRuleContext(AutolevParser.VarTypeContext, 0)

    def varDecl2(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.VarDecl2Context)
        else:
            return self.getTypedRuleContext(AutolevParser.VarDecl2Context, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_varDecl

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterVarDecl'):
            listener.enterVarDecl(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitVarDecl'):
            listener.exitVarDecl(self)