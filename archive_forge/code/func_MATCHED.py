from antlr4 import *
from io import StringIO
import sys
def MATCHED(self):
    return self.getToken(fugue_sqlParser.MATCHED, 0)