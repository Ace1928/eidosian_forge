from antlr4 import *
from io import StringIO
import sys
def BIGINT_LITERAL(self):
    return self.getToken(fugue_sqlParser.BIGINT_LITERAL, 0)