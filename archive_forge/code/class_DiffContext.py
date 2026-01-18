from antlr4 import *
from io import StringIO
import sys
class DiffContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def getRuleIndex(self):
        return AutolevParser.RULE_diff

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterDiff'):
            listener.enterDiff(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitDiff'):
            listener.exitDiff(self)