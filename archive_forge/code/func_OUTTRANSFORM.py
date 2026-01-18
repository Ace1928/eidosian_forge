from antlr4 import *
from io import StringIO
import sys
def OUTTRANSFORM(self):
    return self.getToken(fugue_sqlParser.OUTTRANSFORM, 0)