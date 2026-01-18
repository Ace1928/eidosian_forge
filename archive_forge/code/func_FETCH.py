from antlr4 import *
from io import StringIO
import sys
def FETCH(self):
    return self.getToken(fugue_sqlParser.FETCH, 0)