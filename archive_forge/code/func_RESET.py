from antlr4 import *
from io import StringIO
import sys
def RESET(self):
    return self.getToken(fugue_sqlParser.RESET, 0)