from antlr4 import *
from io import StringIO
import sys
class FugueJsonKeyContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

    def fugueJsonString(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonStringContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueJsonKey

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueJsonKey'):
            return visitor.visitFugueJsonKey(self)
        else:
            return visitor.visitChildren(self)