from antlr4 import *
from io import StringIO
import sys
class InlineTableContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def VALUES(self):
        return self.getToken(fugue_sqlParser.VALUES, 0)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def tableAlias(self):
        return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_inlineTable

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitInlineTable'):
            return visitor.visitInlineTable(self)
        else:
            return visitor.visitChildren(self)