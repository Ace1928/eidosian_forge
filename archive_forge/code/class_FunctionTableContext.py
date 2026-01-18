from antlr4 import *
from io import StringIO
import sys
class FunctionTableContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.funcName = None

    def tableAlias(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

    def errorCapturingIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_functionTable

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFunctionTable'):
            return visitor.visitFunctionTable(self)
        else:
            return visitor.visitChildren(self)