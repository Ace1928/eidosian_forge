from antlr4 import *
from io import StringIO
import sys
class FugueSchemaOpContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueSchemaKey(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaKeyContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaKeyContext, i)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def TILDE(self):
        return self.getToken(fugue_sqlParser.TILDE, 0)

    def PLUS(self):
        return self.getToken(fugue_sqlParser.PLUS, 0)

    def fugueSchema(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSchemaOp

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSchemaOp'):
            return visitor.visitFugueSchemaOp(self)
        else:
            return visitor.visitChildren(self)