from antlr4 import *
from io import StringIO
import sys
def NEQ(self):
    return self.getToken(LaTeXParser.NEQ, 0)