from antlr4 import *
from io import StringIO
import sys
class IntegerLiteralContext(NumberContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIntegerLiteral'):
            return visitor.visitIntegerLiteral(self)
        else:
            return visitor.visitChildren(self)