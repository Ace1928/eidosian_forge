from antlr4 import *
from io import StringIO
import sys
def OR(self):
    return self.getToken(fugue_sqlParser.OR, 0)