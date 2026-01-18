from antlr4 import *
from io import StringIO
import sys
def DIGIT(self, i: int=None):
    if i is None:
        return self.getTokens(LaTeXParser.DIGIT)
    else:
        return self.getToken(LaTeXParser.DIGIT, i)