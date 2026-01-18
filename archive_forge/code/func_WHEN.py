from antlr4 import *
from io import StringIO
import sys
def WHEN(self):
    return self.getToken(fugue_sqlParser.WHEN, 0)