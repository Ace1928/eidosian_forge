from antlr4 import *
from io import StringIO
import sys
class SelectClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self._hint = None
        self.hints = list()

    def SELECT(self):
        return self.getToken(fugue_sqlParser.SELECT, 0)

    def namedExpressionSeq(self):
        return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionSeqContext, 0)

    def setQuantifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.SetQuantifierContext, 0)

    def hint(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.HintContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.HintContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_selectClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSelectClause'):
            return visitor.visitSelectClause(self)
        else:
            return visitor.visitChildren(self)