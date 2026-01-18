from antlr4 import *
from io import StringIO
import sys
class WindowRefContext(WindowSpecContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.name = None
        self.copyFrom(ctx)

    def errorCapturingIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitWindowRef'):
            return visitor.visitWindowRef(self)
        else:
            return visitor.visitChildren(self)