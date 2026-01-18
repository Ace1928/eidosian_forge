from antlr4 import *
from io import StringIO
import sys
def SEED(self):
    return self.getToken(fugue_sqlParser.SEED, 0)