from antlr4 import *
from io import StringIO
import sys
def CHECKPOINT(self):
    return self.getToken(fugue_sqlParser.CHECKPOINT, 0)