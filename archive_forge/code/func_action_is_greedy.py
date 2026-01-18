from argparse import (
from gettext import gettext
from typing import Dict, List, Set, Tuple
def action_is_greedy(action, isoptional=False):
    """Returns True if action will necessarily consume the next argument.
    isoptional indicates whether the argument is an optional (starts with -).
    """
    num_consumed_args = _num_consumed_args.get(action, 0)
    if action.option_strings:
        if not isoptional and (not action_is_satisfied(action)):
            return True
        return action.nargs == REMAINDER
    else:
        return action.nargs == REMAINDER and num_consumed_args >= 1