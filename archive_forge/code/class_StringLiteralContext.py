from antlr4 import *
from io import StringIO
import sys
class StringLiteralContext(ConstantContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def STRING(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.STRING)
        else:
            return self.getToken(fugue_sqlParser.STRING, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitStringLiteral'):
            return visitor.visitStringLiteral(self)
        else:
            return visitor.visitChildren(self)