from antlr4 import *
from io import StringIO
import sys
class AggregationClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self._expression = None
        self.groupingExpressions = list()
        self.kind = None

    def GROUP(self):
        return self.getToken(fugue_sqlParser.GROUP, 0)

    def BY(self):
        return self.getToken(fugue_sqlParser.BY, 0)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def WITH(self):
        return self.getToken(fugue_sqlParser.WITH, 0)

    def SETS(self):
        return self.getToken(fugue_sqlParser.SETS, 0)

    def groupingSet(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.GroupingSetContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.GroupingSetContext, i)

    def ROLLUP(self):
        return self.getToken(fugue_sqlParser.ROLLUP, 0)

    def CUBE(self):
        return self.getToken(fugue_sqlParser.CUBE, 0)

    def GROUPING(self):
        return self.getToken(fugue_sqlParser.GROUPING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_aggregationClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAggregationClause'):
            return visitor.visitAggregationClause(self)
        else:
            return visitor.visitChildren(self)