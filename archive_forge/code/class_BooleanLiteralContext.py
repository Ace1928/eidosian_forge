from antlr4 import *
from io import StringIO
import sys
class BooleanLiteralContext(ConstantContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def booleanValue(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanValueContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitBooleanLiteral'):
            return visitor.visitBooleanLiteral(self)
        else:
            return visitor.visitChildren(self)