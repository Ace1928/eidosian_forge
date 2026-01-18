from antlr4 import *
from io import StringIO
import sys
class AnalyzeContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ANALYZE(self):
        return self.getToken(fugue_sqlParser.ANALYZE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def COMPUTE(self):
        return self.getToken(fugue_sqlParser.COMPUTE, 0)

    def STATISTICS(self):
        return self.getToken(fugue_sqlParser.STATISTICS, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def FOR(self):
        return self.getToken(fugue_sqlParser.FOR, 0)

    def COLUMNS(self):
        return self.getToken(fugue_sqlParser.COLUMNS, 0)

    def identifierSeq(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierSeqContext, 0)

    def ALL(self):
        return self.getToken(fugue_sqlParser.ALL, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAnalyze'):
            return visitor.visitAnalyze(self)
        else:
            return visitor.visitChildren(self)