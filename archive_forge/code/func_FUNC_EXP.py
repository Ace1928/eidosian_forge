from antlr4 import *
from io import StringIO
import sys
def FUNC_EXP(self):
    return self.getToken(LaTeXParser.FUNC_EXP, 0)