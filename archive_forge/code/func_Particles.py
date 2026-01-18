from antlr4 import *
from io import StringIO
import sys
def Particles(self):
    return self.getToken(AutolevParser.Particles, 0)