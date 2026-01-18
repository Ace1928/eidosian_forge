from antlr4 import *
from io import StringIO
import sys
class SingleTableSchemaContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def colTypeList(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

    def EOF(self):
        return self.getToken(fugue_sqlParser.EOF, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_singleTableSchema

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSingleTableSchema'):
            return visitor.visitSingleTableSchema(self)
        else:
            return visitor.visitChildren(self)