from argparse import (
from gettext import gettext
from typing import Dict, List, Set, Tuple
def action_is_satisfied(action):
    """Returns False if the parse would raise an error if no more arguments are given to this action, True otherwise."""
    num_consumed_args = _num_consumed_args.get(action, 0)
    if action.nargs in [OPTIONAL, ZERO_OR_MORE, REMAINDER]:
        return True
    if action.nargs == ONE_OR_MORE:
        return num_consumed_args >= 1
    if action.nargs == PARSER:
        return False
    if action.nargs is None:
        return num_consumed_args == 1
    assert isinstance(action.nargs, int), 'failed to handle a possible nargs value: %r' % action.nargs
    return num_consumed_args == action.nargs