from antlr4 import *
from io import StringIO
import sys
def DOUBLE_LITERAL(self):
    return self.getToken(fugue_sqlParser.DOUBLE_LITERAL, 0)