from antlr4 import *
from io import StringIO
import sys
class SetNamespaceLocationContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def theNamespace(self):
        return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def locationSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSetNamespaceLocation'):
            return visitor.visitSetNamespaceLocation(self)
        else:
            return visitor.visitChildren(self)