from antlr4 import *
from io import StringIO
import sys
def OUT(self):
    return self.getToken(fugue_sqlParser.OUT, 0)