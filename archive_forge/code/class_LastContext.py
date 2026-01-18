from antlr4 import *
from io import StringIO
import sys
class LastContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def LAST(self):
        return self.getToken(fugue_sqlParser.LAST, 0)

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def IGNORE(self):
        return self.getToken(fugue_sqlParser.IGNORE, 0)

    def THENULLS(self):
        return self.getToken(fugue_sqlParser.THENULLS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitLast'):
            return visitor.visitLast(self)
        else:
            return visitor.visitChildren(self)