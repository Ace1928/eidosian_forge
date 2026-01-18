from antlr4 import *
from io import StringIO
import sys
class FugueJsonObjContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueJsonPairs(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonPairsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJsonObj

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJsonObj'):
            return visitor.visitFugueJsonObj(self)
        else:
            return visitor.visitChildren(self)