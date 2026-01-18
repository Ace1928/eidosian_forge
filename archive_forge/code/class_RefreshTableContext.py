from antlr4 import *
from io import StringIO
import sys
class RefreshTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def REFRESH(self):
        return self.getToken(fugue_sqlParser.REFRESH, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRefreshTable'):
            return visitor.visitRefreshTable(self)
        else:
            return visitor.visitChildren(self)