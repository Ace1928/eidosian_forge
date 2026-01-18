from antlr4 import *
from io import StringIO
import sys
def UNDERSCORE(self):
    return self.getToken(LaTeXParser.UNDERSCORE, 0)