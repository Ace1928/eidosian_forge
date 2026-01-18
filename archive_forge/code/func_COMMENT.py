from antlr4 import *
from io import StringIO
import sys
def COMMENT(self):
    return self.getToken(fugue_sqlParser.COMMENT, 0)