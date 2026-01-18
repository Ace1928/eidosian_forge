from antlr4 import *
from io import StringIO
import sys
class FugueRenameExpressionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueRenamePair(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueRenamePairContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueRenamePairContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueRenameExpression

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueRenameExpression'):
            return visitor.visitFugueRenameExpression(self)
        else:
            return visitor.visitChildren(self)