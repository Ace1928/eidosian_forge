from antlr4 import *
from io import StringIO
import sys
def MotionVariables(self):
    return self.getToken(AutolevParser.MotionVariables, 0)