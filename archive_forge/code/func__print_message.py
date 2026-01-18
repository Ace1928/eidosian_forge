import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _print_message(self, message, file=None):
    if message:
        if file is None:
            file = _sys.stderr
        file.write(message)