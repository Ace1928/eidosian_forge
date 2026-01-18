from antlr4 import *
from io import StringIO
import sys
def INTO(self):
    return self.getToken(fugue_sqlParser.INTO, 0)