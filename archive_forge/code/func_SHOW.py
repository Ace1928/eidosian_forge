from antlr4 import *
from io import StringIO
import sys
def SHOW(self):
    return self.getToken(fugue_sqlParser.SHOW, 0)