from antlr4 import *
from io import StringIO
import sys
class CreateTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def createTableHeader(self):
        return self.getTypedRuleContext(fugue_sqlParser.CreateTableHeaderContext, 0)

    def tableProvider(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)

    def createTableClauses(self):
        return self.getTypedRuleContext(fugue_sqlParser.CreateTableClausesContext, 0)

    def colTypeList(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateTable'):
            return visitor.visitCreateTable(self)
        else:
            return visitor.visitChildren(self)