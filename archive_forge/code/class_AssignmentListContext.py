from antlr4 import *
from io import StringIO
import sys
class AssignmentListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def assignment(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.AssignmentContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.AssignmentContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_assignmentList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAssignmentList'):
            return visitor.visitAssignmentList(self)
        else:
            return visitor.visitChildren(self)