from antlr4 import *
from io import StringIO
import sys
class SingleExpressionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def namedExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, 0)

    def EOF(self):
        return self.getToken(fugue_sqlParser.EOF, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_singleExpression

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSingleExpression'):
            return visitor.visitSingleExpression(self)
        else:
            return visitor.visitChildren(self)