import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
class _CountAction(Action):

    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        super(_CountAction, self).__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        new_count = _ensure_value(namespace, self.dest, 0) + 1
        setattr(namespace, self.dest, new_count)