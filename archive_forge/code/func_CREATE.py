from antlr4 import *
from io import StringIO
import sys
def CREATE(self):
    return self.getToken(fugue_sqlParser.CREATE, 0)