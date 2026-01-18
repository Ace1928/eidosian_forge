from antlr4 import *
from io import StringIO
import sys
class PartitionSpecLocationContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def locationSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_partitionSpecLocation

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPartitionSpecLocation'):
            return visitor.visitPartitionSpecLocation(self)
        else:
            return visitor.visitChildren(self)