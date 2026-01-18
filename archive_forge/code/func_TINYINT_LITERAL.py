from antlr4 import *
from io import StringIO
import sys
def TINYINT_LITERAL(self):
    return self.getToken(fugue_sqlParser.TINYINT_LITERAL, 0)