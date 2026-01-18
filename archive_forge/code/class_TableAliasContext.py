from antlr4 import *
from io import StringIO
import sys
class TableAliasContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def strictIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.StrictIdentifierContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def identifierList(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_tableAlias

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTableAlias'):
            return visitor.visitTableAlias(self)
        else:
            return visitor.visitChildren(self)