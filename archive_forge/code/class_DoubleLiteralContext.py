from antlr4 import *
from io import StringIO
import sys
class DoubleLiteralContext(NumberContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def DOUBLE_LITERAL(self):
        return self.getToken(fugue_sqlParser.DOUBLE_LITERAL, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDoubleLiteral'):
            return visitor.visitDoubleLiteral(self)
        else:
            return visitor.visitChildren(self)