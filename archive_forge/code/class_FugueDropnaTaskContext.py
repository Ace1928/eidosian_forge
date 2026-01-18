from antlr4 import *
from io import StringIO
import sys
class FugueDropnaTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.how = None
        self.cols = None
        self.df = None

    def DROP(self):
        return self.getToken(fugue_sqlParser.DROP, 0)

    def ROWS(self):
        return self.getToken(fugue_sqlParser.ROWS, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def THENULLS(self):
        return self.getToken(fugue_sqlParser.THENULLS, 0)

    def ALL(self):
        return self.getToken(fugue_sqlParser.ALL, 0)

    def ANY(self):
        return self.getToken(fugue_sqlParser.ANY, 0)

    def ON(self):
        return self.getToken(fugue_sqlParser.ON, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def fugueCols(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueDropnaTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDropnaTask'):
            return visitor.visitFugueDropnaTask(self)
        else:
            return visitor.visitChildren(self)