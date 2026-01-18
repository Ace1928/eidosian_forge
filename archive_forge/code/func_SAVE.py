from antlr4 import *
from io import StringIO
import sys
def SAVE(self):
    return self.getToken(fugue_sqlParser.SAVE, 0)