from antlr4 import *
from io import StringIO
import sys
def R_BAR(self):
    return self.getToken(LaTeXParser.R_BAR, 0)