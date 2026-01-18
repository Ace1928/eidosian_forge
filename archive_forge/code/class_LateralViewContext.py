from antlr4 import *
from io import StringIO
import sys
class LateralViewContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.tblName = None
        self._identifier = None
        self.colName = list()

    def LATERAL(self):
        return self.getToken(fugue_sqlParser.LATERAL, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def qualifiedName(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

    def identifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

    def OUTER(self):
        return self.getToken(fugue_sqlParser.OUTER, 0)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_lateralView

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitLateralView'):
            return visitor.visitLateralView(self)
        else:
            return visitor.visitChildren(self)