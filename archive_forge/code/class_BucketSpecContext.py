from antlr4 import *
from io import StringIO
import sys
class BucketSpecContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def CLUSTERED(self):
        return self.getToken(fugue_sqlParser.CLUSTERED, 0)

    def BY(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.BY)
        else:
            return self.getToken(fugue_sqlParser.BY, i)

    def identifierList(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

    def INTO(self):
        return self.getToken(fugue_sqlParser.INTO, 0)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def BUCKETS(self):
        return self.getToken(fugue_sqlParser.BUCKETS, 0)

    def SORTED(self):
        return self.getToken(fugue_sqlParser.SORTED, 0)

    def orderedIdentifierList(self):
        return self.getTypedRuleContext(fugue_sqlParser.OrderedIdentifierListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_bucketSpec

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitBucketSpec'):
            return visitor.visitBucketSpec(self)
        else:
            return visitor.visitChildren(self)