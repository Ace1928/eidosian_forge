from antlr4 import *
from io import StringIO
import sys
def SYSTEM(self):
    return self.getToken(fugue_sqlParser.SYSTEM, 0)