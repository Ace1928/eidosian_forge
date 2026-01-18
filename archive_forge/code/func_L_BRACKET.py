from antlr4 import *
from io import StringIO
import sys
def L_BRACKET(self):
    return self.getToken(LaTeXParser.L_BRACKET, 0)