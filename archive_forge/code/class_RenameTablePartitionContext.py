from antlr4 import *
from io import StringIO
import sys
class RenameTablePartitionContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.ifrom = None
        self.to = None
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def RENAME(self):
        return self.getToken(fugue_sqlParser.RENAME, 0)

    def TO(self):
        return self.getToken(fugue_sqlParser.TO, 0)

    def partitionSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.PartitionSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRenameTablePartition'):
            return visitor.visitRenameTablePartition(self)
        else:
            return visitor.visitChildren(self)