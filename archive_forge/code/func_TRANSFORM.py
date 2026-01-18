from antlr4 import *
from io import StringIO
import sys
def TRANSFORM(self):
    return self.getToken(fugue_sqlParser.TRANSFORM, 0)