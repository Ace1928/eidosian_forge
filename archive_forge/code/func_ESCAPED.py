from antlr4 import *
from io import StringIO
import sys
def ESCAPED(self):
    return self.getToken(fugue_sqlParser.ESCAPED, 0)