from antlr4 import *
from io import StringIO
import sys
class SingleStatementContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def statement(self):
        return self.getTypedRuleContext(fugue_sqlParser.StatementContext, 0)

    def EOF(self):
        return self.getToken(fugue_sqlParser.EOF, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_singleStatement

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSingleStatement'):
            return visitor.visitSingleStatement(self)
        else:
            return visitor.visitChildren(self)