from antlr4 import *
from io import StringIO
import sys
def PRINT(self):
    return self.getToken(fugue_sqlParser.PRINT, 0)