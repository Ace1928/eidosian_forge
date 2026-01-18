import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
class _StoreConstAction(Action):

    def __init__(self, option_strings, dest, const, default=None, required=False, help=None, metavar=None):
        super(_StoreConstAction, self).__init__(option_strings=option_strings, dest=dest, nargs=0, const=const, default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)