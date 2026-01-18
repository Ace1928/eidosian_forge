from antlr4 import *
from io import StringIO
import sys
def TABLE(self):
    return self.getToken(fugue_sqlParser.TABLE, 0)