from antlr4 import *
from io import StringIO
import sys
class SampleContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def TABLESAMPLE(self):
        return self.getToken(fugue_sqlParser.TABLESAMPLE, 0)

    def sampleMethod(self):
        return self.getTypedRuleContext(fugue_sqlParser.SampleMethodContext, 0)

    def SYSTEM(self):
        return self.getToken(fugue_sqlParser.SYSTEM, 0)

    def BERNOULLI(self):
        return self.getToken(fugue_sqlParser.BERNOULLI, 0)

    def RESERVOIR(self):
        return self.getToken(fugue_sqlParser.RESERVOIR, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_sample

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSample'):
            return visitor.visitSample(self)
        else:
            return visitor.visitChildren(self)