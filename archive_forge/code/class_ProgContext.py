from antlr4 import *
from io import StringIO
import sys
class ProgContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def stat(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(AutolevParser.StatContext)
        else:
            return self.getTypedRuleContext(AutolevParser.StatContext, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_prog

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterProg'):
            listener.enterProg(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitProg'):
            listener.exitProg(self)