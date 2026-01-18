from antlr4 import *
from io import StringIO
import sys
class FugueJsonContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueJsonValue(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonValueContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJson

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJson'):
            return visitor.visitFugueJson(self)
        else:
            return visitor.visitChildren(self)