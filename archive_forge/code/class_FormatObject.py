import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
class FormatObject:

    def is_compose(self):
        return False

    def is_choice(self):
        return False

    def is_indent(self):
        return False

    def is_string(self):
        return False

    def is_linebreak(self):
        return False

    def is_nil(self):
        return True

    def children(self):
        return []

    def as_tuple(self):
        return None

    def space_upto_nl(self):
        return (0, False)

    def flat(self):
        return self