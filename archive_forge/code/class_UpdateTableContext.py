from antlr4 import *
from io import StringIO
import sys
class UpdateTableContext(DmlStatementNoWithContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def UPDATE(self):
        return self.getToken(fugue_sqlParser.UPDATE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def tableAlias(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

    def setClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.SetClauseContext, 0)

    def whereClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitUpdateTable'):
            return visitor.visitUpdateTable(self)
        else:
            return visitor.visitChildren(self)