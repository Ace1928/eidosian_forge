from antlr4 import *
from io import StringIO
import sys
def FILEFORMAT(self):
    return self.getToken(fugue_sqlParser.FILEFORMAT, 0)