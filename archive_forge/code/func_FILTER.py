from antlr4 import *
from io import StringIO
import sys
def FILTER(self):
    return self.getToken(fugue_sqlParser.FILTER, 0)