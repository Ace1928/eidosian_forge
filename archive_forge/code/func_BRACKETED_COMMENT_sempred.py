from antlr4 import *
from io import StringIO
import sys
def BRACKETED_COMMENT_sempred(self, localctx: RuleContext, predIndex: int):
    if predIndex == 5:
        return not self.isHint()