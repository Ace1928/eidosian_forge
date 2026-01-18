from antlr4 import *
from io import StringIO
import sys
class NumericLiteralContext(ConstantContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def number(self):
        return self.getTypedRuleContext(fugue_sqlParser.NumberContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNumericLiteral'):
            return visitor.visitNumericLiteral(self)
        else:
            return visitor.visitChildren(self)