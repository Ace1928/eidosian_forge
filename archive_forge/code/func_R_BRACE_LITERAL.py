from antlr4 import *
from io import StringIO
import sys
def R_BRACE_LITERAL(self):
    return self.getToken(LaTeXParser.R_BRACE_LITERAL, 0)