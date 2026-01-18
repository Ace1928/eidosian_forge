from antlr4 import *
from io import StringIO
import sys
def FORMAT(self):
    return self.getToken(fugue_sqlParser.FORMAT, 0)