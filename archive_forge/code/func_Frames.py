from antlr4 import *
from io import StringIO
import sys
def Frames(self):
    return self.getToken(AutolevParser.Frames, 0)