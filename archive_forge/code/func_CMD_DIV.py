from antlr4 import *
from io import StringIO
import sys
def CMD_DIV(self):
    return self.getToken(LaTeXParser.CMD_DIV, 0)