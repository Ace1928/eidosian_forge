from antlr4 import *
from io import StringIO
import sys
class TablePropertyKeyContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_tablePropertyKey

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTablePropertyKey'):
            return visitor.visitTablePropertyKey(self)
        else:
            return visitor.visitChildren(self)