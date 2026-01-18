from antlr4 import *
from io import StringIO
import sys
class CommentTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.comment = None
        self.copyFrom(ctx)

    def COMMENT(self):
        return self.getToken(fugue_sqlParser.COMMENT, 0)

    def ON(self):
        return self.getToken(fugue_sqlParser.ON, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def IS(self):
        return self.getToken(fugue_sqlParser.IS, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCommentTable'):
            return visitor.visitCommentTable(self)
        else:
            return visitor.visitChildren(self)