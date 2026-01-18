from antlr4 import *
from io import StringIO
import sys
def R_FLOOR(self):
    return self.getToken(LaTeXParser.R_FLOOR, 0)