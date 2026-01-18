from antlr4 import *
from io import StringIO
import sys
def PRECEDING(self):
    return self.getToken(fugue_sqlParser.PRECEDING, 0)