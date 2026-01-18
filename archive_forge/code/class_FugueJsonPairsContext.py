from antlr4 import *
from io import StringIO
import sys
class FugueJsonPairsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueJsonPair(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueJsonPairContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonPairContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJsonPairs

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJsonPairs'):
            return visitor.visitFugueJsonPairs(self)
        else:
            return visitor.visitChildren(self)