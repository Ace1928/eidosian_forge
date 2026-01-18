from antlr4 import *
from io import StringIO
import sys
def LOCAL(self):
    return self.getToken(fugue_sqlParser.LOCAL, 0)