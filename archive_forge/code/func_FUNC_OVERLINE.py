from antlr4 import *
from io import StringIO
import sys
def FUNC_OVERLINE(self):
    return self.getToken(LaTeXParser.FUNC_OVERLINE, 0)