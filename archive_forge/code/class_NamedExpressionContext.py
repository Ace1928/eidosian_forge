from antlr4 import *
from io import StringIO
import sys
class NamedExpressionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.name = None

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def identifierList(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def errorCapturingIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_namedExpression

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNamedExpression'):
            return visitor.visitNamedExpression(self)
        else:
            return visitor.visitChildren(self)