from antlr4 import *
from io import StringIO
import sys
class RepairTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def MSCK(self):
        return self.getToken(fugue_sqlParser.MSCK, 0)

    def REPAIR(self):
        return self.getToken(fugue_sqlParser.REPAIR, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitRepairTable'):
            return visitor.visitRepairTable(self)
        else:
            return visitor.visitChildren(self)