from antlr4 import *
from io import StringIO
import sys
def L_ANGLE(self):
    return self.getToken(LaTeXParser.L_ANGLE, 0)