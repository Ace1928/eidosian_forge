from antlr4 import *
from io import StringIO
import sys
class RecoverPartitionsContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def RECOVER(self):
        return self.getToken(fugue_sqlParser.RECOVER, 0)

    def PARTITIONS(self):
        return self.getToken(fugue_sqlParser.PARTITIONS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRecoverPartitions'):
            return visitor.visitRecoverPartitions(self)
        else:
            return visitor.visitChildren(self)