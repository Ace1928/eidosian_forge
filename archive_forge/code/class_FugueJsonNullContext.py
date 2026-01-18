from antlr4 import *
from io import StringIO
import sys
class FugueJsonNullContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJsonNull

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJsonNull'):
            return visitor.visitFugueJsonNull(self)
        else:
            return visitor.visitChildren(self)