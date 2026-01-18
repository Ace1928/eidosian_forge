from antlr4 import *
from io import StringIO
import sys
class FuguePartitionNumContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fuguePartitionNumber(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionNumberContext, 0)

    def fuguePartitionNum(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FuguePartitionNumContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionNumContext, i)

    def PLUS(self):
        return self.getToken(fugue_sqlParser.PLUS, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def ASTERISK(self):
        return self.getToken(fugue_sqlParser.ASTERISK, 0)

    def SLASH(self):
        return self.getToken(fugue_sqlParser.SLASH, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fuguePartitionNum

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFuguePartitionNum'):
            return visitor.visitFuguePartitionNum(self)
        else:
            return visitor.visitChildren(self)