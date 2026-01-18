from antlr4 import *
from io import StringIO
import sys
class MatchedActionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def DELETE(self):
        return self.getToken(fugue_sqlParser.DELETE, 0)

    def UPDATE(self):
        return self.getToken(fugue_sqlParser.UPDATE, 0)

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def ASTERISK(self):
        return self.getToken(fugue_sqlParser.ASTERISK, 0)

    def assignmentList(self):
        return self.getTypedRuleContext(fugue_sqlParser.AssignmentListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_matchedAction

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitMatchedAction'):
            return visitor.visitMatchedAction(self)
        else:
            return visitor.visitChildren(self)