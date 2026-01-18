from antlr4 import *
from io import StringIO
import sys
def TAKE(self):
    return self.getToken(fugue_sqlParser.TAKE, 0)