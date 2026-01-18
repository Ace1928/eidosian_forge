from antlr4 import *
from io import StringIO
import sys
def SINGLE_QUOTES(self):
    return self.getToken(LaTeXParser.SINGLE_QUOTES, 0)