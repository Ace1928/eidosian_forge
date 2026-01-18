from antlr4 import *
from io import StringIO
import sys
class DescribeRelationContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.option = None
        self.copyFrom(ctx)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def DESC(self):
        return self.getToken(fugue_sqlParser.DESC, 0)

    def DESCRIBE(self):
        return self.getToken(fugue_sqlParser.DESCRIBE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def describeColName(self):
        return self.getTypedRuleContext(fugue_sqlParser.DescribeColNameContext, 0)

    def EXTENDED(self):
        return self.getToken(fugue_sqlParser.EXTENDED, 0)

    def FORMATTED(self):
        return self.getToken(fugue_sqlParser.FORMATTED, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitDescribeRelation'):
            return visitor.visitDescribeRelation(self)
        else:
            return visitor.visitChildren(self)