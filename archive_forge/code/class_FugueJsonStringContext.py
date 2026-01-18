from antlr4 import *
from io import StringIO
import sys
class FugueJsonStringContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJsonString

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJsonString'):
            return visitor.visitFugueJsonString(self)
        else:
            return visitor.visitChildren(self)