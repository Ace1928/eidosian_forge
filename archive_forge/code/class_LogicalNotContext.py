from antlr4 import *
from io import StringIO
import sys
class LogicalNotContext(BooleanExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def booleanExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitLogicalNot'):
            return visitor.visitLogicalNot(self)
        else:
            return visitor.visitChildren(self)