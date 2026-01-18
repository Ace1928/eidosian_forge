from antlr4 import *
from io import StringIO
import sys
class ComplexColTypeListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def complexColType(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ComplexColTypeContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ComplexColTypeContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_complexColTypeList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitComplexColTypeList'):
            return visitor.visitComplexColTypeList(self)
        else:
            return visitor.visitChildren(self)