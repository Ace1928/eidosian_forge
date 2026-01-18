from antlr4 import *
from io import StringIO
import sys
class HiveChangeColumnContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.table = None
        self.colName = None
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def CHANGE(self):
        return self.getToken(fugue_sqlParser.CHANGE, 0)

    def colType(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColTypeContext, 0)

    def multipartIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def COLUMN(self):
        return self.getToken(fugue_sqlParser.COLUMN, 0)

    def colPosition(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColPositionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitHiveChangeColumn'):
            return visitor.visitHiveChangeColumn(self)
        else:
            return visitor.visitChildren(self)