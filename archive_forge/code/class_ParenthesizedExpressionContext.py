from antlr4 import *
from io import StringIO
import sys
class ParenthesizedExpressionContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitParenthesizedExpression'):
            return visitor.visitParenthesizedExpression(self)
        else:
            return visitor.visitChildren(self)