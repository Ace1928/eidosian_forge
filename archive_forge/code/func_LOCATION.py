from antlr4 import *
from io import StringIO
import sys
def LOCATION(self):
    return self.getToken(fugue_sqlParser.LOCATION, 0)