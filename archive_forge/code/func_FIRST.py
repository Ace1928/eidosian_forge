from antlr4 import *
from io import StringIO
import sys
def FIRST(self):
    return self.getToken(fugue_sqlParser.FIRST, 0)