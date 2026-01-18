from antlr4 import *
from io import StringIO
import sys
class ColPositionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.position = None
        self.afterCol = None

    def FIRST(self):
        return self.getToken(fugue_sqlParser.FIRST, 0)

    def AFTER(self):
        return self.getToken(fugue_sqlParser.AFTER, 0)

    def errorCapturingIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_colPosition

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitColPosition'):
            return visitor.visitColPosition(self)
        else:
            return visitor.visitChildren(self)