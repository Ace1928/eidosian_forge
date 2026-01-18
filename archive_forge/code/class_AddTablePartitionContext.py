from antlr4 import *
from io import StringIO
import sys
class AddTablePartitionContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def ADD(self):
        return self.getToken(fugue_sqlParser.ADD, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def partitionSpecLocation(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.PartitionSpecLocationContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecLocationContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAddTablePartition'):
            return visitor.visitAddTablePartition(self)
        else:
            return visitor.visitChildren(self)