from antlr4 import *
from io import StringIO
import sys
def IMPORT(self):
    return self.getToken(fugue_sqlParser.IMPORT, 0)