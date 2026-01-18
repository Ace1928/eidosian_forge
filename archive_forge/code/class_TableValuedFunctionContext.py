from antlr4 import *
from io import StringIO
import sys
class TableValuedFunctionContext(RelationPrimaryContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def functionTable(self):
        return self.getTypedRuleContext(fugue_sqlParser.FunctionTableContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTableValuedFunction'):
            return visitor.visitTableValuedFunction(self)
        else:
            return visitor.visitChildren(self)