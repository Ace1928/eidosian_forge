import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _get_action_name(argument):
    if argument is None:
        return None
    elif argument.option_strings:
        return '/'.join(argument.option_strings)
    elif argument.metavar not in (None, SUPPRESS):
        return argument.metavar
    elif argument.dest not in (None, SUPPRESS):
        return argument.dest
    else:
        return None