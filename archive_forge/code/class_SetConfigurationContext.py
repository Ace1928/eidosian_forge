from antlr4 import *
from io import StringIO
import sys
class SetConfigurationContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSetConfiguration'):
            return visitor.visitSetConfiguration(self)
        else:
            return visitor.visitChildren(self)