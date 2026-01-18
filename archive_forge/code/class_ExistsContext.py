from antlr4 import *
from io import StringIO
import sys
class ExistsContext(BooleanExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitExists'):
            return visitor.visitExists(self)
        else:
            return visitor.visitChildren(self)