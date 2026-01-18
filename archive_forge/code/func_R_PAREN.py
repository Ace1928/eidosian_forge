from antlr4 import *
from io import StringIO
import sys
def R_PAREN(self):
    return self.getToken(LaTeXParser.R_PAREN, 0)