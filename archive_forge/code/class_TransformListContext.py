from antlr4 import *
from io import StringIO
import sys
class TransformListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self._transform = None
        self.transforms = list()

    def transform(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TransformContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TransformContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_transformList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTransformList'):
            return visitor.visitTransformList(self)
        else:
            return visitor.visitChildren(self)