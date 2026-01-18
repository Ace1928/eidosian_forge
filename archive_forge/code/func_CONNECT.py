from antlr4 import *
from io import StringIO
import sys
def CONNECT(self):
    return self.getToken(fugue_sqlParser.CONNECT, 0)