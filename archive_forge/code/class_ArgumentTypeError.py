import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
class ArgumentTypeError(Exception):
    """An error from trying to convert a command line string to a type."""
    pass