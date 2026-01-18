from antlr4 import *
from io import StringIO
import sys
class FugueExtensionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.domain = None

    def fugueIdentifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueIdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueExtension

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueExtension'):
            return visitor.visitFugueExtension(self)
        else:
            return visitor.visitChildren(self)