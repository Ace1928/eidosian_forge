from antlr4 import *
from io import StringIO
import sys
class FugueIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueIdentifier'):
            return visitor.visitFugueIdentifier(self)
        else:
            return visitor.visitChildren(self)