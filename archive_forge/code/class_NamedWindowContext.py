from antlr4 import *
from io import StringIO
import sys
class NamedWindowContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.name = None

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def windowSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.WindowSpecContext, 0)

    def errorCapturingIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_namedWindow

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNamedWindow'):
            return visitor.visitNamedWindow(self)
        else:
            return visitor.visitChildren(self)