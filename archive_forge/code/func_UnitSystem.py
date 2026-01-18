from antlr4 import *
from io import StringIO
import sys
def UnitSystem(self):
    return self.getToken(AutolevParser.UnitSystem, 0)