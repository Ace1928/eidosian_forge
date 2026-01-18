from antlr4 import *
from io import StringIO
import sys
def LETTER_sempred(self, localctx: RuleContext, predIndex: int):
    if predIndex == 4:
        return not self.allUpperCase