from antlr4 import *
from io import StringIO
import sys
def BAR(self):
    return self.getToken(LaTeXParser.BAR, 0)