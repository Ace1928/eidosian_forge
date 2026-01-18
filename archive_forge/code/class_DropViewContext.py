from antlr4 import *
from io import StringIO
import sys
class DropViewContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def DROP(self):
        return self.getToken(fugue_sqlParser.DROP, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDropView'):
            return visitor.visitDropView(self)
        else:
            return visitor.visitChildren(self)