import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _metavar_formatter(self, action, default_metavar):
    if action.metavar is not None:
        result = action.metavar
    elif action.choices is not None:
        choice_strs = [str(choice) for choice in action.choices]
        result = '{%s}' % ','.join(choice_strs)
    else:
        result = default_metavar

    def format(tuple_size):
        if isinstance(result, tuple):
            return result
        else:
            return (result,) * tuple_size
    return format