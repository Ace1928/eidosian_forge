from antlr4 import *
from io import StringIO
import sys
class FugueFileFormatContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def PARQUET(self):
        return self.getToken(fugue_sqlParser.PARQUET, 0)

    def CSV(self):
        return self.getToken(fugue_sqlParser.CSV, 0)

    def JSON(self):
        return self.getToken(fugue_sqlParser.JSON, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueFileFormat

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueFileFormat'):
            return visitor.visitFugueFileFormat(self)
        else:
            return visitor.visitChildren(self)