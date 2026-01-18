from antlr4 import *
from io import StringIO
import sys
def Bodies(self):
    return self.getToken(AutolevParser.Bodies, 0)