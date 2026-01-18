from antlr4 import *
from io import StringIO
import sys
def GRANT(self):
    return self.getToken(fugue_sqlParser.GRANT, 0)