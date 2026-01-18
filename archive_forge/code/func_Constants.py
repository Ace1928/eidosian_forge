from antlr4 import *
from io import StringIO
import sys
def Constants(self):
    return self.getToken(AutolevParser.Constants, 0)