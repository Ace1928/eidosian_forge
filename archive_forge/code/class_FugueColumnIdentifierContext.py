from antlr4 import *
from io import StringIO
import sys
class FugueColumnIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueColumnIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueColumnIdentifier'):
            return visitor.visitFugueColumnIdentifier(self)
        else:
            return visitor.visitChildren(self)