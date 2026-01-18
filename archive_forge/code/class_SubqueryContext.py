from antlr4 import *
from io import StringIO
import sys
class SubqueryContext(QueryPrimaryContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def query(self):
        return self.getTypedRuleContext(sqlParser.QueryContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSubquery'):
            return visitor.visitSubquery(self)
        else:
            return visitor.visitChildren(self)