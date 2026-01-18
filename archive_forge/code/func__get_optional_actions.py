import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _get_optional_actions(self):
    return [action for action in self._actions if action.option_strings]