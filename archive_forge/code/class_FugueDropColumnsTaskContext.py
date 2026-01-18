from antlr4 import *
from io import StringIO
import sys
class FugueDropColumnsTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.cols = None
        self.df = None

    def DROP(self):
        return self.getToken(fugue_sqlParser.DROP, 0)

    def COLUMNS(self):
        return self.getToken(fugue_sqlParser.COLUMNS, 0)

    def fugueCols(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueDropColumnsTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDropColumnsTask'):
            return visitor.visitFugueDropColumnsTask(self)
        else:
            return visitor.visitChildren(self)