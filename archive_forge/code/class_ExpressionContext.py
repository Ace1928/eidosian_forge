from antlr4 import *
from io import StringIO
import sys
class ExpressionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def booleanExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_expression

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitExpression'):
            return visitor.visitExpression(self)
        else:
            return visitor.visitChildren(self)