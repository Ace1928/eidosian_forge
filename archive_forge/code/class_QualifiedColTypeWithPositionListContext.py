from antlr4 import *
from io import StringIO
import sys
class QualifiedColTypeWithPositionListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def qualifiedColTypeWithPosition(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.QualifiedColTypeWithPositionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedColTypeWithPositionContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_qualifiedColTypeWithPositionList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQualifiedColTypeWithPositionList'):
            return visitor.visitQualifiedColTypeWithPositionList(self)
        else:
            return visitor.visitChildren(self)