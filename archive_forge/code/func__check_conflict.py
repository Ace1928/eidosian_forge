import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _check_conflict(self, action):
    confl_optionals = []
    for option_string in action.option_strings:
        if option_string in self._option_string_actions:
            confl_optional = self._option_string_actions[option_string]
            confl_optionals.append((option_string, confl_optional))
    if confl_optionals:
        conflict_handler = self._get_handler()
        conflict_handler(action, confl_optionals)