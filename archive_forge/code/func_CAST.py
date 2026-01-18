from antlr4 import *
from io import StringIO
import sys
def CAST(self):
    return self.getToken(fugue_sqlParser.CAST, 0)