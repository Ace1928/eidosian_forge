from antlr4 import *
from io import StringIO
import sys
class PartitionSpecContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def PARTITION(self):
        return self.getToken(fugue_sqlParser.PARTITION, 0)

    def partitionVal(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.PartitionValContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.PartitionValContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_partitionSpec

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPartitionSpec'):
            return visitor.visitPartitionSpec(self)
        else:
            return visitor.visitChildren(self)