from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class ShellTokenType(enum.Enum):
    ARG = 1
    FLAG = 2
    TERMINATOR = 3
    IO = 4
    REDIRECTION = 5
    FILE = 6
    TRAILING_BACKSLASH = 7