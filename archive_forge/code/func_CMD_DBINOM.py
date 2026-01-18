from antlr4 import *
from io import StringIO
import sys
def CMD_DBINOM(self):
    return self.getToken(LaTeXParser.CMD_DBINOM, 0)