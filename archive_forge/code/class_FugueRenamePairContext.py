from antlr4 import *
from io import StringIO
import sys
class FugueRenamePairContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.key = None
        self.value = None

    def fugueSchemaKey(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaKeyContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaKeyContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueRenamePair

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueRenamePair'):
            return visitor.visitFugueRenamePair(self)
        else:
            return visitor.visitChildren(self)