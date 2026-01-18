import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _get_optional_kwargs(self, *args, **kwargs):
    option_strings = []
    long_option_strings = []
    for option_string in args:
        if not option_string[0] in self.prefix_chars:
            msg = _('invalid option string %r: must start with a character %r')
            tup = (option_string, self.prefix_chars)
            raise ValueError(msg % tup)
        option_strings.append(option_string)
        if option_string[0] in self.prefix_chars:
            if len(option_string) > 1:
                if option_string[1] in self.prefix_chars:
                    long_option_strings.append(option_string)
    dest = kwargs.pop('dest', None)
    if dest is None:
        if long_option_strings:
            dest_option_string = long_option_strings[0]
        else:
            dest_option_string = option_strings[0]
        dest = dest_option_string.lstrip(self.prefix_chars)
        if not dest:
            msg = _('dest= is required for options like %r')
            raise ValueError(msg % option_string)
        dest = dest.replace('-', '_')
    return dict(kwargs, dest=dest, option_strings=option_strings)