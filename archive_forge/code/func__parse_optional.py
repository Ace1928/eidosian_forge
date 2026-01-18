import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _parse_optional(self, arg_string):
    if not arg_string:
        return None
    if not arg_string[0] in self.prefix_chars:
        return None
    if arg_string in self._option_string_actions:
        action = self._option_string_actions[arg_string]
        return (action, arg_string, None)
    if len(arg_string) == 1:
        return None
    if '=' in arg_string:
        option_string, explicit_arg = arg_string.split('=', 1)
        if option_string in self._option_string_actions:
            action = self._option_string_actions[option_string]
            return (action, option_string, explicit_arg)
    option_tuples = self._get_option_tuples(arg_string)
    if len(option_tuples) > 1:
        options = ', '.join([option_string for action, option_string, explicit_arg in option_tuples])
        tup = (arg_string, options)
        self.error(_('ambiguous option: %s could match %s') % tup)
    elif len(option_tuples) == 1:
        option_tuple, = option_tuples
        return option_tuple
    if self._negative_number_matcher.match(arg_string):
        if not self._has_negative_number_optionals:
            return None
    if ' ' in arg_string:
        return None
    return (None, arg_string, None)