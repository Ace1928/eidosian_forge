from antlr4 import *
from io import StringIO
import sys
def USING(self):
    return self.getToken(fugue_sqlParser.USING, 0)