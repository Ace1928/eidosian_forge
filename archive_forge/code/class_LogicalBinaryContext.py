from antlr4 import *
from io import StringIO
import sys
class LogicalBinaryContext(BooleanExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.left = None
        self.theOperator = None
        self.right = None
        self.copyFrom(ctx)

    def booleanExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.BooleanExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, i)

    def AND(self):
        return self.getToken(fugue_sqlParser.AND, 0)

    def OR(self):
        return self.getToken(fugue_sqlParser.OR, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitLogicalBinary'):
            return visitor.visitLogicalBinary(self)
        else:
            return visitor.visitChildren(self)