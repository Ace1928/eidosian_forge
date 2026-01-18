from antlr4 import *
from io import StringIO
import sys
def SMALLINT_LITERAL(self):
    return self.getToken(fugue_sqlParser.SMALLINT_LITERAL, 0)