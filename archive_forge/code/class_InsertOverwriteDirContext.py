from antlr4 import *
from io import StringIO
import sys
class InsertOverwriteDirContext(InsertIntoContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.path = None
        self.options = None
        self.copyFrom(ctx)

    def INSERT(self):
        return self.getToken(fugue_sqlParser.INSERT, 0)

    def OVERWRITE(self):
        return self.getToken(fugue_sqlParser.OVERWRITE, 0)

    def DIRECTORY(self):
        return self.getToken(fugue_sqlParser.DIRECTORY, 0)

    def tableProvider(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)

    def LOCAL(self):
        return self.getToken(fugue_sqlParser.LOCAL, 0)

    def OPTIONS(self):
        return self.getToken(fugue_sqlParser.OPTIONS, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def tablePropertyList(self):
        return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitInsertOverwriteDir'):
            return visitor.visitInsertOverwriteDir(self)
        else:
            return visitor.visitChildren(self)