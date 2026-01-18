from antlr4 import *
from io import StringIO
import sys
class FugueCreateTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.params = None

    def CREATE(self):
        return self.getToken(fugue_sqlParser.CREATE, 0)

    def fugueSingleOutputExtensionCommon(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSingleOutputExtensionCommonContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueCreateTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueCreateTask'):
            return visitor.visitFugueCreateTask(self)
        else:
            return visitor.visitChildren(self)