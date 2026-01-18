import sys
import os
import getopt
from pyparsing import *
class ToInteger(TokenConverter):
    """Converter to make token into an integer."""

    def postParse(self, instring, loc, tokenlist):
        return int(tokenlist[0])