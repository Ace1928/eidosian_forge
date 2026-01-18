from antlr4 import *
from io import StringIO
import sys
class FugueWildSchemaContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueWildSchemaPair(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueWildSchemaPairContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueWildSchemaPairContext, i)

    def fugueSchemaOp(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaOpContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaOpContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueWildSchema

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueWildSchema'):
            return visitor.visitFugueWildSchema(self)
        else:
            return visitor.visitChildren(self)