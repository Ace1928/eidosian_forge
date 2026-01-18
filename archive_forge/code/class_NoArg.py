import functools
import re
import sys
from Xlib.support import lock
class NoArg(Option):
    """Value is provided to constructor."""

    def __init__(self, specifier, value):
        self.specifier = specifier
        self.value = value

    def parse(self, name, db, args):
        db.insert(name + self.specifier, self.value)
        return args[1:]