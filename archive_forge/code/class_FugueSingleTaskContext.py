from antlr4 import *
from io import StringIO
import sys
class FugueSingleTaskContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueNestableTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueNestableTaskContext, 0)

    def fugueOutputTransformTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueOutputTransformTaskContext, 0)

    def fugueOutputTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueOutputTaskContext, 0)

    def fuguePrintTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FuguePrintTaskContext, 0)

    def fugueSaveTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSaveTaskContext, 0)

    def fugueModuleTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueModuleTaskContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSingleTask

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSingleTask'):
            return visitor.visitFugueSingleTask(self)
        else:
            return visitor.visitChildren(self)