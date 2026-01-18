from antlr4 import *
from io import StringIO
import sys
class UseContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def USE(self):
        return self.getToken(fugue_sqlParser.USE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def NAMESPACE(self):
        return self.getToken(fugue_sqlParser.NAMESPACE, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitUse'):
            return visitor.visitUse(self)
        else:
            return visitor.visitChildren(self)