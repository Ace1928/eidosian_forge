from antlr4 import *
from io import StringIO
import sys
def DOUBLE_LITERAL_sempred(self, localctx: RuleContext, predIndex: int):
    if predIndex == 2:
        return self.isValidDecimal