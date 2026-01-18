import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _add_action(self, action):
    if action.option_strings:
        self._optionals._add_action(action)
    else:
        self._positionals._add_action(action)
    return action