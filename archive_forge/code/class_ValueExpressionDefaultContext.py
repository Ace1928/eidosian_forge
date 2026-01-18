from antlr4 import *
from io import StringIO
import sys
class ValueExpressionDefaultContext(ValueExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def primaryExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitValueExpressionDefault'):
            return visitor.visitValueExpressionDefault(self)
        else:
            return visitor.visitChildren(self)