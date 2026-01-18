from antlr4 import *
from io import StringIO
import sys
class ErrorIdentContext(ErrorCapturingIdentifierExtraContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def MINUS(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.MINUS)
        else:
            return self.getToken(fugue_sqlParser.MINUS, i)

    def identifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitErrorIdent'):
            return visitor.visitErrorIdent(self)
        else:
            return visitor.visitChildren(self)