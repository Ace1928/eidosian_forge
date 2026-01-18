from antlr4 import *
from io import StringIO
import sys
class FunctionNameContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def qualifiedName(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

    def FILTER(self):
        return self.getToken(fugue_sqlParser.FILTER, 0)

    def LEFT(self):
        return self.getToken(fugue_sqlParser.LEFT, 0)

    def RIGHT(self):
        return self.getToken(fugue_sqlParser.RIGHT, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_functionName

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFunctionName'):
            return visitor.visitFunctionName(self)
        else:
            return visitor.visitChildren(self)