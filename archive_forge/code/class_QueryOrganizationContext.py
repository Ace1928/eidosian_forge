from antlr4 import *
from io import StringIO
import sys
class QueryOrganizationContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self._sortItem = None
        self.order = list()
        self._expression = None
        self.clusterBy = list()
        self.distributeBy = list()
        self.sort = list()
        self.limit = None

    def ORDER(self):
        return self.getToken(fugue_sqlParser.ORDER, 0)

    def BY(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.BY)
        else:
            return self.getToken(fugue_sqlParser.BY, i)

    def CLUSTER(self):
        return self.getToken(fugue_sqlParser.CLUSTER, 0)

    def DISTRIBUTE(self):
        return self.getToken(fugue_sqlParser.DISTRIBUTE, 0)

    def SORT(self):
        return self.getToken(fugue_sqlParser.SORT, 0)

    def windowClause(self):
        return self.getTypedRuleContext(fugue_sqlParser.WindowClauseContext, 0)

    def LIMIT(self):
        return self.getToken(fugue_sqlParser.LIMIT, 0)

    def sortItem(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.SortItemContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.SortItemContext, i)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def ALL(self):
        return self.getToken(fugue_sqlParser.ALL, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_queryOrganization

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQueryOrganization'):
            return visitor.visitQueryOrganization(self)
        else:
            return visitor.visitChildren(self)