from antlr4 import *
from io import StringIO
import sys
def CMD_TIMES(self):
    return self.getToken(LaTeXParser.CMD_TIMES, 0)