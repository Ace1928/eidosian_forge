from antlr4 import *
from io import StringIO
import sys
class BigIntLiteralContext(NumberContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def BIGINT_LITERAL(self):
        return self.getToken(fugue_sqlParser.BIGINT_LITERAL, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitBigIntLiteral'):
            return visitor.visitBigIntLiteral(self)
        else:
            return visitor.visitChildren(self)