from antlr4 import *
from io import StringIO
import sys
def EXPLAIN(self):
    return self.getToken(fugue_sqlParser.EXPLAIN, 0)