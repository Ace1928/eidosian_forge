from antlr4 import *
from io import StringIO
import sys
def BACKQUOTED_IDENTIFIER(self):
    return self.getToken(fugue_sqlParser.BACKQUOTED_IDENTIFIER, 0)