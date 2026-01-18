from antlr4 import *
from io import StringIO
import sys
class FugueProcessTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.dfs = None
        self.partition = None
        self.params = None

    def PROCESS(self):
        return self.getToken(fugue_sqlParser.PROCESS, 0)

    def fugueSingleOutputExtensionCommon(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSingleOutputExtensionCommonContext, 0)

    def fugueDataFrames(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

    def fuguePrepartition(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueProcessTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueProcessTask'):
            return visitor.visitFugueProcessTask(self)
        else:
            return visitor.visitChildren(self)