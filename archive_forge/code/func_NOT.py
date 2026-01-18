from antlr4 import *
from io import StringIO
import sys
def NOT(self):
    return self.getToken(fugue_sqlParser.NOT, 0)