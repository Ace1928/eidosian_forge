from antlr4 import *
from io import StringIO
import sys
def DISTINCT(self):
    return self.getToken(fugue_sqlParser.DISTINCT, 0)