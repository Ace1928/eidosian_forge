from antlr4 import *
from io import StringIO
import sys
class SingleTableIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def tableIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, 0)

    def EOF(self):
        return self.getToken(fugue_sqlParser.EOF, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_singleTableIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSingleTableIdentifier'):
            return visitor.visitSingleTableIdentifier(self)
        else:
            return visitor.visitChildren(self)