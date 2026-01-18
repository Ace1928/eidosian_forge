from antlr4 import *
from io import StringIO
import sys
def LIST(self):
    return self.getToken(fugue_sqlParser.LIST, 0)