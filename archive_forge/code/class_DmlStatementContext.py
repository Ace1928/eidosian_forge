from antlr4 import *
from io import StringIO
import sys
class DmlStatementContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def dmlStatementNoWith(self):
        return self.getTypedRuleContext(fugue_sqlParser.DmlStatementNoWithContext, 0)

    def ctes(self):
        return self.getTypedRuleContext(fugue_sqlParser.CtesContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDmlStatement'):
            return visitor.visitDmlStatement(self)
        else:
            return visitor.visitChildren(self)