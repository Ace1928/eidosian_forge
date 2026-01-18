from antlr4 import *
from io import StringIO
import sys
class CodegenContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ID(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.ID)
        else:
            return self.getToken(AutolevParser.ID, i)

    def functionCall(self):
        return self.getTypedRuleContext(AutolevParser.FunctionCallContext, 0)

    def matrixInOutput(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.MatrixInOutputContext)
        else:
            return self.getTypedRuleContext(AutolevParser.MatrixInOutputContext, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_codegen

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterCodegen'):
            listener.enterCodegen(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitCodegen'):
            listener.exitCodegen(self)