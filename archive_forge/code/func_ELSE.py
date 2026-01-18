from antlr4 import *
from io import StringIO
import sys
def ELSE(self):
    return self.getToken(fugue_sqlParser.ELSE, 0)