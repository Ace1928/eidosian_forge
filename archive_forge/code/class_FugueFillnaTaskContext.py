from antlr4 import *
from io import StringIO
import sys
class FugueFillnaTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.params = None
        self.df = None

    def FILL(self):
        return self.getToken(fugue_sqlParser.FILL, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def THENULLS(self):
        return self.getToken(fugue_sqlParser.THENULLS, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueFillnaTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueFillnaTask'):
            return visitor.visitFugueFillnaTask(self)
        else:
            return visitor.visitChildren(self)