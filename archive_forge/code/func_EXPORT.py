from antlr4 import *
from io import StringIO
import sys
def EXPORT(self):
    return self.getToken(fugue_sqlParser.EXPORT, 0)