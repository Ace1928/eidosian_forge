import functools
import re
import sys
from Xlib.support import lock
class SepArg(Option):
    """Value is the next argument."""

    def __init__(self, specifier):
        self.specifier = specifier

    def parse(self, name, db, args):
        db.insert(name + self.specifier, args[1])
        return args[2:]