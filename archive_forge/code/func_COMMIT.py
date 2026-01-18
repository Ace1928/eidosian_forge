from antlr4 import *
from io import StringIO
import sys
def COMMIT(self):
    return self.getToken(fugue_sqlParser.COMMIT, 0)