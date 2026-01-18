import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _join_parts(self, part_strings):
    return ''.join([part for part in part_strings if part and part is not SUPPRESS])