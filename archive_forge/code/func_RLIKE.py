from antlr4 import *
from io import StringIO
import sys
def RLIKE(self):
    return self.getToken(fugue_sqlParser.RLIKE, 0)