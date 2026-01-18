import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _get_help_string(self, action):
    help = action.help
    if '%(default)' not in action.help:
        if action.default is not SUPPRESS:
            defaulting_nargs = [OPTIONAL, ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                help += ' (default: %(default)s)'
    return help