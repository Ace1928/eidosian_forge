from antlr4 import *
from io import StringIO
import sys
class FugueSchemaPairContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.key = None
        self.value = None

    def fugueSchemaKey(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaKeyContext, 0)

    def fugueSchemaType(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaTypeContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSchemaPair

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSchemaPair'):
            return visitor.visitFugueSchemaPair(self)
        else:
            return visitor.visitChildren(self)