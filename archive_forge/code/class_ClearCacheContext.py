from antlr4 import *
from io import StringIO
import sys
class ClearCacheContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def CLEAR(self):
        return self.getToken(fugue_sqlParser.CLEAR, 0)

    def CACHE(self):
        return self.getToken(fugue_sqlParser.CACHE, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitClearCache'):
            return visitor.visitClearCache(self)
        else:
            return visitor.visitChildren(self)