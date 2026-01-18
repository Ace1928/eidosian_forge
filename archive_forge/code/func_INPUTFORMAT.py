from antlr4 import *
from io import StringIO
import sys
def INPUTFORMAT(self):
    return self.getToken(fugue_sqlParser.INPUTFORMAT, 0)