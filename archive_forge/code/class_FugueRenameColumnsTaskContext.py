from antlr4 import *
from io import StringIO
import sys
class FugueRenameColumnsTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.cols = None
        self.df = None

    def RENAME(self):
        return self.getToken(fugue_sqlParser.RENAME, 0)

    def COLUMNS(self):
        return self.getToken(fugue_sqlParser.COLUMNS, 0)

    def fugueRenameExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueRenameExpressionContext, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def fugueDataFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueRenameColumnsTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueRenameColumnsTask'):
            return visitor.visitFugueRenameColumnsTask(self)
        else:
            return visitor.visitChildren(self)