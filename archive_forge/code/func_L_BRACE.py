from antlr4 import *
from io import StringIO
import sys
def L_BRACE(self):
    return self.getToken(LaTeXParser.L_BRACE, 0)