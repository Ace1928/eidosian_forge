from antlr4 import *
from io import StringIO
import sys
class InsertOverwriteTableContext(InsertIntoContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def INSERT(self):
        return self.getToken(fugue_sqlParser.INSERT, 0)

    def OVERWRITE(self):
        return self.getToken(fugue_sqlParser.OVERWRITE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitInsertOverwriteTable'):
            return visitor.visitInsertOverwriteTable(self)
        else:
            return visitor.visitChildren(self)