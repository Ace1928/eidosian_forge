from antlr4 import *
from io import StringIO
import sys
class ConstantDefaultContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def constant(self):
        return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitConstantDefault'):
            return visitor.visitConstantDefault(self)
        else:
            return visitor.visitChildren(self)