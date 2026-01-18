import functools
import re
import sys
from Xlib.support import lock
class SkipNArgs(Option):
    """Ignore this option and the next COUNT arguments."""

    def __init__(self, count):
        self.count = count

    def parse(self, name, db, args):
        return args[1 + self.count:]