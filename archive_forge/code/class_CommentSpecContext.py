from antlr4 import *
from io import StringIO
import sys
class CommentSpecContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def COMMENT(self):
        return self.getToken(fugue_sqlParser.COMMENT, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_commentSpec

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCommentSpec'):
            return visitor.visitCommentSpec(self)
        else:
            return visitor.visitChildren(self)