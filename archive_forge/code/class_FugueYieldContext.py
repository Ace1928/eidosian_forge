from antlr4 import *
from io import StringIO
import sys
class FugueYieldContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.name = None

    def YIELD(self):
        return self.getToken(fugue_sqlParser.YIELD, 0)

    def FILE(self):
        return self.getToken(fugue_sqlParser.FILE, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def fugueIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

    def DATAFRAME(self):
        return self.getToken(fugue_sqlParser.DATAFRAME, 0)

    def LOCAL(self):
        return self.getToken(fugue_sqlParser.LOCAL, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueYield

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueYield'):
            return visitor.visitFugueYield(self)
        else:
            return visitor.visitChildren(self)