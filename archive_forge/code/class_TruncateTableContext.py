from antlr4 import *
from io import StringIO
import sys
class TruncateTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def TRUNCATE(self):
        return self.getToken(fugue_sqlParser.TRUNCATE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTruncateTable'):
            return visitor.visitTruncateTable(self)
        else:
            return visitor.visitChildren(self)