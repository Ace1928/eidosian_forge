from antlr4 import *
from io import StringIO
import sys
def NAMESPACE(self):
    return self.getToken(fugue_sqlParser.NAMESPACE, 0)