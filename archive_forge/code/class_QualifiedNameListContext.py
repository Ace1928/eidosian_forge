from antlr4 import *
from io import StringIO
import sys
class QualifiedNameListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def qualifiedName(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.QualifiedNameContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_qualifiedNameList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitQualifiedNameList'):
            return visitor.visitQualifiedNameList(self)
        else:
            return visitor.visitChildren(self)