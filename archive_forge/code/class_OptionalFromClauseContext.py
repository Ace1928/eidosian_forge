from antlr4 import *
from io import StringIO
import sys
class OptionalFromClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fromClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.FromClauseContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_optionalFromClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitOptionalFromClause'):
            return visitor.visitOptionalFromClause(self)
        else:
            return visitor.visitChildren(self)