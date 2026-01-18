from antlr4 import *
from io import StringIO
import sys
def CMD_BINOM(self):
    return self.getToken(LaTeXParser.CMD_BINOM, 0)