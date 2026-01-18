from antlr4 import *
from io import StringIO
import sys
def TITLE(self):
    return self.getToken(fugue_sqlParser.TITLE, 0)