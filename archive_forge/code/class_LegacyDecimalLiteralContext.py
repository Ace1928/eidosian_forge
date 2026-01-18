from antlr4 import *
from io import StringIO
import sys
class LegacyDecimalLiteralContext(NumberContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def EXPONENT_VALUE(self):
        return self.getToken(fugue_sqlParser.EXPONENT_VALUE, 0)

    def DECIMAL_VALUE(self):
        return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitLegacyDecimalLiteral'):
            return visitor.visitLegacyDecimalLiteral(self)
        else:
            return visitor.visitChildren(self)