from antlr4 import *
from io import StringIO
import sys
class ShowColumnsContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.table = None
        self.ns = None
        self.copyFrom(ctx)

    def SHOW(self):
        return self.getToken(fugue_sqlParser.SHOW, 0)

    def COLUMNS(self):
        return self.getToken(fugue_sqlParser.COLUMNS, 0)

    def FROM(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.FROM)
        else:
            return self.getToken(fugue_sqlParser.FROM, i)

    def IN(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.IN)
        else:
            return self.getToken(fugue_sqlParser.IN, i)

    def multipartIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitShowColumns'):
            return visitor.visitShowColumns(self)
        else:
            return visitor.visitChildren(self)