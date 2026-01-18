from antlr4 import *
from io import StringIO
import sys
class ConstantListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def constant(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ConstantContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_constantList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitConstantList'):
            return visitor.visitConstantList(self)
        else:
            return visitor.visitChildren(self)