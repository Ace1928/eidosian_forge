import os
import warnings
from IPython.utils.ipstruct import Struct
class InputTermColors:
    """Color escape sequences for input prompts.

    This class is similar to TermColors, but the escapes are wrapped in \\001
    and \\002 so that readline can properly know the length of each line and
    can wrap lines accordingly.  Use this class for any colored text which
    needs to be used in input prompts, such as in calls to raw_input().

    This class defines the escape sequences for all the standard (ANSI?)
    colors in terminals. Also defines a NoColor escape which is just the null
    string, suitable for defining 'dummy' color schemes in terminals which get
    confused by color escapes.

    This class should be used as a mixin for building color schemes."""
    NoColor = ''
    if os.name == 'nt' and os.environ.get('TERM', 'dumb') == 'emacs':
        Normal = '\x1b[0m'
        _base = '\x1b[%sm'
    else:
        Normal = '\x01\x1b[0m\x02'
        _base = '\x01\x1b[%sm\x02'