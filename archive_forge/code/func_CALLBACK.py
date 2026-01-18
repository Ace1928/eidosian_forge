from antlr4 import *
from io import StringIO
import sys
def CALLBACK(self):
    return self.getToken(fugue_sqlParser.CALLBACK, 0)