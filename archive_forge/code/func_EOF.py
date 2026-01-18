from antlr4 import *
from io import StringIO
import sys
def EOF(self):
    return self.getToken(fugue_sqlParser.EOF, 0)