from antlr4 import *
from io import StringIO
import sys
class PrimaryExpressionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_primaryExpression

    def copyFrom(self, ctx: ParserRuleContext):
        super().copyFrom(ctx)