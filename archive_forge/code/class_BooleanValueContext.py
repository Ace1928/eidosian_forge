from antlr4 import *
from io import StringIO
import sys
class BooleanValueContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def TRUE(self):
        return self.getToken(fugue_sqlParser.TRUE, 0)

    def FALSE(self):
        return self.getToken(fugue_sqlParser.FALSE, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_booleanValue

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitBooleanValue'):
            return visitor.visitBooleanValue(self)
        else:
            return visitor.visitChildren(self)