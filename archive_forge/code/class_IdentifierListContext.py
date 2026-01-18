from antlr4 import *
from io import StringIO
import sys
class IdentifierListContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifierSeq(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierSeqContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_identifierList

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIdentifierList'):
            return visitor.visitIdentifierList(self)
        else:
            return visitor.visitChildren(self)