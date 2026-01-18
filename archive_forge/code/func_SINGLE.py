from antlr4 import *
from io import StringIO
import sys
def SINGLE(self):
    return self.getToken(fugue_sqlParser.SINGLE, 0)