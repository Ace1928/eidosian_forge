from antlr4 import *
from io import StringIO
import sys
class AlterColumnActionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.setOrDrop = None

    def TYPE(self):
        return self.getToken(fugue_sqlParser.TYPE, 0)

    def dataType(self):
        return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

    def commentSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

    def colPosition(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColPositionContext, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def DROP(self):
        return self.getToken(fugue_sqlParser.DROP, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_alterColumnAction

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAlterColumnAction'):
            return visitor.visitAlterColumnAction(self)
        else:
            return visitor.visitChildren(self)