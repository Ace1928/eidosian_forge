from antlr4 import *
from io import StringIO
import sys
class FugueZipTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.dfs = None
        self.how = None
        self.by = None
        self.presort = None

    def ZIP(self):
        return self.getToken(fugue_sqlParser.ZIP, 0)

    def fugueDataFrames(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

    def BY(self):
        return self.getToken(fugue_sqlParser.BY, 0)

    def PRESORT(self):
        return self.getToken(fugue_sqlParser.PRESORT, 0)

    def fugueZipType(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueZipTypeContext, 0)

    def fugueCols(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

    def fugueColsSort(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueColsSortContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueZipTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueZipTask'):
            return visitor.visitFugueZipTask(self)
        else:
            return visitor.visitChildren(self)