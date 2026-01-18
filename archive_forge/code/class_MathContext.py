from antlr4 import *
from io import StringIO
import sys
class MathContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def relation(self):
        return self.getTypedRuleContext(LaTeXParser.RelationContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_math