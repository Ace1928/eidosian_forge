from antlr4 import *
from io import StringIO
import sys
def USER(self):
    return self.getToken(fugue_sqlParser.USER, 0)