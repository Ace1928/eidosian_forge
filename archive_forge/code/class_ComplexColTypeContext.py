from antlr4 import *
from io import StringIO
import sys
class ComplexColTypeContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def dataType(self):
        return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

    def NOT(self):
        return self.getToken(fugue_sqlParser.NOT, 0)

    def THENULL(self):
        return self.getToken(fugue_sqlParser.THENULL, 0)

    def commentSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_complexColType

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitComplexColType'):
            return visitor.visitComplexColType(self)
        else:
            return visitor.visitChildren(self)