from antlr4 import *
from io import StringIO
import sys
def ROLE(self):
    return self.getToken(fugue_sqlParser.ROLE, 0)