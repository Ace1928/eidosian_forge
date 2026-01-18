from antlr4 import *
from io import StringIO
import sys
def ROWCOUNT(self):
    return self.getToken(fugue_sqlParser.ROWCOUNT, 0)