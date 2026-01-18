from antlr4 import *
from io import StringIO
import sys
class ResourceContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_resource

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitResource'):
            return visitor.visitResource(self)
        else:
            return visitor.visitChildren(self)