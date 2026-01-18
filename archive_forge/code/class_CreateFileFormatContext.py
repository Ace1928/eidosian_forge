from antlr4 import *
from io import StringIO
import sys
class CreateFileFormatContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def STORED(self):
        return self.getToken(fugue_sqlParser.STORED, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def fileFormat(self):
        return self.getTypedRuleContext(fugue_sqlParser.FileFormatContext, 0)

    def BY(self):
        return self.getToken(fugue_sqlParser.BY, 0)

    def storageHandler(self):
        return self.getTypedRuleContext(fugue_sqlParser.StorageHandlerContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_createFileFormat

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateFileFormat'):
            return visitor.visitCreateFileFormat(self)
        else:
            return visitor.visitChildren(self)