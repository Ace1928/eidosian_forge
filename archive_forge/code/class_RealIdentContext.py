from antlr4 import *
from io import StringIO
import sys
class RealIdentContext(ErrorCapturingIdentifierExtraContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRealIdent'):
            return visitor.visitRealIdent(self)
        else:
            return visitor.visitChildren(self)