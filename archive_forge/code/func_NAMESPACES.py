from antlr4 import *
from io import StringIO
import sys
def NAMESPACES(self):
    return self.getToken(fugue_sqlParser.NAMESPACES, 0)