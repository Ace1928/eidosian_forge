from antlr4 import *
from io import StringIO
import sys
def FLOAT(self):
    return self.getToken(AutolevParser.FLOAT, 0)