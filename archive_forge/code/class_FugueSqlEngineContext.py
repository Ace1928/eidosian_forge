from antlr4 import *
from io import StringIO
import sys
class FugueSqlEngineContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.fugueUsing = None
        self.params = None

    def CONNECT(self):
        return self.getToken(fugue_sqlParser.CONNECT, 0)

    def fugueExtension(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSqlEngine

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSqlEngine'):
            return visitor.visitFugueSqlEngine(self)
        else:
            return visitor.visitChildren(self)