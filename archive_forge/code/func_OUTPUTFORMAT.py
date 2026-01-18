from antlr4 import *
from io import StringIO
import sys
def OUTPUTFORMAT(self):
    return self.getToken(fugue_sqlParser.OUTPUTFORMAT, 0)