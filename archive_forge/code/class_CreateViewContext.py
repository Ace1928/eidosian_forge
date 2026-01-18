from antlr4 import *
from io import StringIO
import sys
class CreateViewContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def OR(self):
        return self.getToken(fugue_sqlParser.OR, 0)

    def REPLACE(self):
        return self.getToken(fugue_sqlParser.REPLACE, 0)

    def TEMPORARY(self):
        return self.getToken(fugue_sqlParser.TEMPORARY, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def identifierCommentList(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierCommentListContext, 0)

    def commentSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

    def PARTITIONED(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.PARTITIONED)
        else:
            return self.getToken(fugue_sqlParser.PARTITIONED, i)

    def ON(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.ON)
        else:
            return self.getToken(fugue_sqlParser.ON, i)

    def identifierList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IdentifierListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, i)

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

    def GLOBAL(self):
        return self.getToken(fugue_sqlParser.GLOBAL, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateView'):
            return visitor.visitCreateView(self)
        else:
            return visitor.visitChildren(self)