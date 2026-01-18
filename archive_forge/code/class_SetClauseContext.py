from antlr4 import *
from io import StringIO
import sys
class SetClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def assignmentList(self):
        return self.getTypedRuleContext(fugue_sqlParser.AssignmentListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_setClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSetClause'):
            return visitor.visitSetClause(self)
        else:
            return visitor.visitChildren(self)