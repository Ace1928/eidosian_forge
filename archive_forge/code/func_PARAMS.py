from antlr4 import *
from io import StringIO
import sys
def PARAMS(self):
    return self.getToken(fugue_sqlParser.PARAMS, 0)