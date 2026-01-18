from antlr4 import *
from io import StringIO
import sys
class FrameBoundContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.boundType = None

    def UNBOUNDED(self):
        return self.getToken(fugue_sqlParser.UNBOUNDED, 0)

    def PRECEDING(self):
        return self.getToken(fugue_sqlParser.PRECEDING, 0)

    def FOLLOWING(self):
        return self.getToken(fugue_sqlParser.FOLLOWING, 0)

    def ROW(self):
        return self.getToken(fugue_sqlParser.ROW, 0)

    def CURRENT(self):
        return self.getToken(fugue_sqlParser.CURRENT, 0)

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_frameBound

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFrameBound'):
            return visitor.visitFrameBound(self)
        else:
            return visitor.visitChildren(self)