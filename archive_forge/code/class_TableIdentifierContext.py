from antlr4 import *
from io import StringIO
import sys
class TableIdentifierContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.db = None
        self.table = None

    def errorCapturingIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ErrorCapturingIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_tableIdentifier

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTableIdentifier'):
            return visitor.visitTableIdentifier(self)
        else:
            return visitor.visitChildren(self)