from antlr4 import *
from io import StringIO
import sys
def SLASH(self):
    return self.getToken(fugue_sqlParser.SLASH, 0)