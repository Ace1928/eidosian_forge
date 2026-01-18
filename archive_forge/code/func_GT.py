from antlr4 import *
from io import StringIO
import sys
def GT(self):
    return self.getToken(LaTeXParser.GT, 0)