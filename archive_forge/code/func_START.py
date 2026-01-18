from antlr4 import *
from io import StringIO
import sys
def START(self):
    return self.getToken(fugue_sqlParser.START, 0)