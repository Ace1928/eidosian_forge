from antlr4 import *
from io import StringIO
import sys
class FromStatementContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fromClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.FromClauseContext, 0)

    def fromStatementBody(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FromStatementBodyContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FromStatementBodyContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fromStatement

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFromStatement'):
            return visitor.visitFromStatement(self)
        else:
            return visitor.visitChildren(self)