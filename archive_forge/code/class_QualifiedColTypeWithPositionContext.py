from antlr4 import *
from io import StringIO
import sys
class QualifiedColTypeWithPositionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.name = None

    def dataType(self):
        return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def commentSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

    def colPosition(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColPositionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_qualifiedColTypeWithPosition

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQualifiedColTypeWithPosition'):
            return visitor.visitQualifiedColTypeWithPosition(self)
        else:
            return visitor.visitChildren(self)