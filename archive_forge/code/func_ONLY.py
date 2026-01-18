from antlr4 import *
from io import StringIO
import sys
def ONLY(self):
    return self.getToken(fugue_sqlParser.ONLY, 0)