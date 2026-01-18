from antlr4 import *
from io import StringIO
import sys
class UnquotedIdentifierContext(StrictIdentifierContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def IDENTIFIER(self):
        return self.getToken(fugue_sqlParser.IDENTIFIER, 0)

    def nonReserved(self):
        return self.getTypedRuleContext(fugue_sqlParser.NonReservedContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitUnquotedIdentifier'):
            return visitor.visitUnquotedIdentifier(self)
        else:
            return visitor.visitChildren(self)