from antlr4 import *
from io import StringIO
import sys
def COLUMNS(self):
    return self.getToken(fugue_sqlParser.COLUMNS, 0)