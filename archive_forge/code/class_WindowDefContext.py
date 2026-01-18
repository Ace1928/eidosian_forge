from antlr4 import *
from io import StringIO
import sys
class WindowDefContext(WindowSpecContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self._expression = None
        self.partition = list()
        self.copyFrom(ctx)

    def CLUSTER(self):
        return self.getToken(fugue_sqlParser.CLUSTER, 0)

    def BY(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.BY)
        else:
            return self.getToken(fugue_sqlParser.BY, i)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def windowFrame(self):
        return self.getTypedRuleContext(fugue_sqlParser.WindowFrameContext, 0)

    def sortItem(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.SortItemContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.SortItemContext, i)

    def PARTITION(self):
        return self.getToken(fugue_sqlParser.PARTITION, 0)

    def DISTRIBUTE(self):
        return self.getToken(fugue_sqlParser.DISTRIBUTE, 0)

    def ORDER(self):
        return self.getToken(fugue_sqlParser.ORDER, 0)

    def SORT(self):
        return self.getToken(fugue_sqlParser.SORT, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitWindowDef'):
            return visitor.visitWindowDef(self)
        else:
            return visitor.visitChildren(self)