from antlr4 import *
from io import StringIO
import sys
def LAZY(self):
    return self.getToken(fugue_sqlParser.LAZY, 0)