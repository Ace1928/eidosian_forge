from antlr4 import *
from io import StringIO
import sys
def FIELDS(self):
    return self.getToken(fugue_sqlParser.FIELDS, 0)