from antlr4 import *
from io import StringIO
import sys
def UNION(self):
    return self.getToken(fugue_sqlParser.UNION, 0)