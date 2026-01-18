from antlr4 import *
from io import StringIO
import sys
class DescribeNamespaceContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def theNamespace(self):
        return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def DESC(self):
        return self.getToken(fugue_sqlParser.DESC, 0)

    def DESCRIBE(self):
        return self.getToken(fugue_sqlParser.DESCRIBE, 0)

    def EXTENDED(self):
        return self.getToken(fugue_sqlParser.EXTENDED, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDescribeNamespace'):
            return visitor.visitDescribeNamespace(self)
        else:
            return visitor.visitChildren(self)