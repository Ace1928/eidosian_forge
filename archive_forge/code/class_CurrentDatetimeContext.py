from antlr4 import *
from io import StringIO
import sys
class CurrentDatetimeContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.name = None
        self.copyFrom(ctx)

    def CURRENT_DATE(self):
        return self.getToken(fugue_sqlParser.CURRENT_DATE, 0)

    def CURRENT_TIMESTAMP(self):
        return self.getToken(fugue_sqlParser.CURRENT_TIMESTAMP, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCurrentDatetime'):
            return visitor.visitCurrentDatetime(self)
        else:
            return visitor.visitChildren(self)