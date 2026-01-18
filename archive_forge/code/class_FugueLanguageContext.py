from antlr4 import *
from io import StringIO
import sys
class FugueLanguageContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def EOF(self):
        return self.getToken(fugue_sqlParser.EOF, 0)

    def fugueSingleTask(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueSingleTaskContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleTaskContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueLanguage

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueLanguage'):
            return visitor.visitFugueLanguage(self)
        else:
            return visitor.visitChildren(self)