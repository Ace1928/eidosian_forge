from antlr4 import *
from io import StringIO
import sys
def EXTRACT(self):
    return self.getToken(fugue_sqlParser.EXTRACT, 0)