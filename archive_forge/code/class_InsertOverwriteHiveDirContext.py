from antlr4 import *
from io import StringIO
import sys
class InsertOverwriteHiveDirContext(InsertIntoContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.path = None
        self.copyFrom(ctx)

    def INSERT(self):
        return self.getToken(fugue_sqlParser.INSERT, 0)

    def OVERWRITE(self):
        return self.getToken(fugue_sqlParser.OVERWRITE, 0)

    def DIRECTORY(self):
        return self.getToken(fugue_sqlParser.DIRECTORY, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def LOCAL(self):
        return self.getToken(fugue_sqlParser.LOCAL, 0)

    def rowFormat(self):
        return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, 0)

    def createFileFormat(self):
        return self.getTypedRuleContext(fugue_sqlParser.CreateFileFormatContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitInsertOverwriteHiveDir'):
            return visitor.visitInsertOverwriteHiveDir(self)
        else:
            return visitor.visitChildren(self)