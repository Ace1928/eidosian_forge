from antlr4 import *
from io import StringIO
import sys
class Outputs2Context(ParserRuleContext):
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
        return AutolevParser.RULE_outputs2

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterOutputs2'):
            listener.enterOutputs2(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitOutputs2'):
            listener.exitOutputs2(self)