from antlr4 import *
from io import StringIO
import sys
def OPTION(self):
    return self.getToken(fugue_sqlParser.OPTION, 0)