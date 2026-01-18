from antlr4 import *
from io import StringIO
import sys
class ResetConfigurationContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def RESET(self):
        return self.getToken(fugue_sqlParser.RESET, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitResetConfiguration'):
            return visitor.visitResetConfiguration(self)
        else:
            return visitor.visitChildren(self)