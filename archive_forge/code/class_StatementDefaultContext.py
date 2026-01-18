from antlr4 import *
from io import StringIO
import sys
class StatementDefaultContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitStatementDefault'):
            return visitor.visitStatementDefault(self)
        else:
            return visitor.visitChildren(self)