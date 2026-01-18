from antlr4 import *
from io import StringIO
import sys
def PERCENTLIT(self):
    return self.getToken(fugue_sqlParser.PERCENTLIT, 0)