from antlr4 import *
from io import StringIO
import sys
def COLON(self):
    return self.getToken(LaTeXParser.COLON, 0)