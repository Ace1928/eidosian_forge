from antlr4 import *
from io import StringIO
import sys
class RangesContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def INT(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.INT)
        else:
            return self.getToken(AutolevParser.INT, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_ranges

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterRanges'):
            listener.enterRanges(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitRanges'):
            listener.exitRanges(self)