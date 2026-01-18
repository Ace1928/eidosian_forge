from antlr4 import *
from io import StringIO
import sys
class FugueTransformTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.dfs = None
        self.partition = None
        self.params = None
        self.callback = None

    def TRANSFORM(self):
        return self.getToken(fugue_sqlParser.TRANSFORM, 0)

    def fugueSingleOutputExtensionCommonWild(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSingleOutputExtensionCommonWildContext, 0)

    def CALLBACK(self):
        return self.getToken(fugue_sqlParser.CALLBACK, 0)

    def fugueDataFrames(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

    def fuguePrepartition(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

    def fugueExtension(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueTransformTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueTransformTask'):
            return visitor.visitFugueTransformTask(self)
        else:
            return visitor.visitChildren(self)