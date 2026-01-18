from antlr4 import *
from io import StringIO
import sys
class FugueNestableTaskCollectionNoSelectContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def fugueTransformTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueTransformTaskContext, 0)

    def fugueProcessTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueProcessTaskContext, 0)

    def fugueZipTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueZipTaskContext, 0)

    def fugueCreateTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueCreateTaskContext, 0)

    def fugueCreateDataTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueCreateDataTaskContext, 0)

    def fugueLoadTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueLoadTaskContext, 0)

    def fugueSaveAndUseTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSaveAndUseTaskContext, 0)

    def fugueRenameColumnsTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueRenameColumnsTaskContext, 0)

    def fugueAlterColumnsTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueAlterColumnsTaskContext, 0)

    def fugueDropColumnsTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDropColumnsTaskContext, 0)

    def fugueDropnaTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueDropnaTaskContext, 0)

    def fugueFillnaTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueFillnaTaskContext, 0)

    def fugueSampleTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSampleTaskContext, 0)

    def fugueTakeTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueTakeTaskContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueNestableTaskCollectionNoSelect

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueNestableTaskCollectionNoSelect'):
            return visitor.visitFugueNestableTaskCollectionNoSelect(self)
        else:
            return visitor.visitChildren(self)