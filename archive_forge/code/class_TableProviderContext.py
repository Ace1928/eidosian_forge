from antlr4 import *
from io import StringIO
import sys
class TableProviderContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_tableProvider

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTableProvider'):
            return visitor.visitTableProvider(self)
        else:
            return visitor.visitChildren(self)