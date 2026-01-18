from antlr4 import *
from io import StringIO
import sys
class CreateTableLikeContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.target = None
        self.source = None
        self.tableProps = None
        self.copyFrom(ctx)

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def LIKE(self):
        return self.getToken(fugue_sqlParser.LIKE, 0)

    def tableIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TableIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, i)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def tableProvider(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TableProviderContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, i)

    def rowFormat(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.RowFormatContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, i)

    def createFileFormat(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.CreateFileFormatContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.CreateFileFormatContext, i)

    def locationSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

    def TBLPROPERTIES(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.TBLPROPERTIES)
        else:
            return self.getToken(fugue_sqlParser.TBLPROPERTIES, i)

    def tablePropertyList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateTableLike'):
            return visitor.visitCreateTableLike(self)
        else:
            return visitor.visitChildren(self)