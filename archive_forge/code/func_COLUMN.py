from antlr4 import *
from io import StringIO
import sys
def COLUMN(self):
    return self.getToken(fugue_sqlParser.COLUMN, 0)