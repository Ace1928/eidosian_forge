from antlr4 import *
from io import StringIO
import sys
class QuotedIdentifierAlternativeContext(StrictIdentifierContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def quotedIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.QuotedIdentifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQuotedIdentifierAlternative'):
            return visitor.visitQuotedIdentifierAlternative(self)
        else:
            return visitor.visitChildren(self)