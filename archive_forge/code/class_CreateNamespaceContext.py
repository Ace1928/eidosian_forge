from antlr4 import *
from io import StringIO
import sys
class CreateNamespaceContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def theNamespace(self):
        return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def commentSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

    def locationSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

    def WITH(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.WITH)
        else:
            return self.getToken(fugue_sqlParser.WITH, i)

    def tablePropertyList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

    def DBPROPERTIES(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.DBPROPERTIES)
        else:
            return self.getToken(fugue_sqlParser.DBPROPERTIES, i)

    def PROPERTIES(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.PROPERTIES)
        else:
            return self.getToken(fugue_sqlParser.PROPERTIES, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateNamespace'):
            return visitor.visitCreateNamespace(self)
        else:
            return visitor.visitChildren(self)