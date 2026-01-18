from antlr4 import *
from io import StringIO
import sys
def R_BRACE(self):
    return self.getToken(LaTeXParser.R_BRACE, 0)