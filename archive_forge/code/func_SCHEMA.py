from antlr4 import *
from io import StringIO
import sys
def SCHEMA(self):
    return self.getToken(fugue_sqlParser.SCHEMA, 0)