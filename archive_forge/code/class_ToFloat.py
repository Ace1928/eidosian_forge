import sys
import os
import getopt
from pyparsing import *
class ToFloat(TokenConverter):
    """Converter to make token into a float."""

    def postParse(self, instring, loc, tokenlist):
        return float(tokenlist[0])