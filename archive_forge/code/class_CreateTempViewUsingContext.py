from antlr4 import *
from io import StringIO
import sys
class CreateTempViewUsingContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def TEMPORARY(self):
        return self.getToken(fugue_sqlParser.TEMPORARY, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def tableIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, 0)

    def tableProvider(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)

    def OR(self):
        return self.getToken(fugue_sqlParser.OR, 0)

    def REPLACE(self):
        return self.getToken(fugue_sqlParser.REPLACE, 0)

    def GLOBAL(self):
        return self.getToken(fugue_sqlParser.GLOBAL, 0)

    def colTypeList(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

    def OPTIONS(self):
        return self.getToken(fugue_sqlParser.OPTIONS, 0)

    def tablePropertyList(self):
        return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateTempViewUsing'):
            return visitor.visitCreateTempViewUsing(self)
        else:
            return visitor.visitChildren(self)