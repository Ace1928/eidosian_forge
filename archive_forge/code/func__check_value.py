import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _check_value(self, action, value):
    if action.choices is not None and value not in action.choices:
        tup = (value, ', '.join(map(repr, action.choices)))
        msg = _('invalid choice: %r (choose from %s)') % tup
        raise ArgumentError(action, msg)