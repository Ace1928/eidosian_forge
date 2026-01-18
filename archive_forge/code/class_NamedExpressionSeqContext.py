from antlr4 import *
from io import StringIO
import sys
class NamedExpressionSeqContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def namedExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.NamedExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_namedExpressionSeq

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNamedExpressionSeq'):
            return visitor.visitNamedExpressionSeq(self)
        else:
            return visitor.visitChildren(self)