from antlr4 import *
from io import StringIO
import sys
class FugueDataFramePairContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.key = None
        self.value = None

    def fugueIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

    def EQUAL(self):
        return self.getToken(fugue_sqlParser.EQUAL, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueDataFramePair

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDataFramePair'):
            return visitor.visitFugueDataFramePair(self)
        else:
            return visitor.visitChildren(self)