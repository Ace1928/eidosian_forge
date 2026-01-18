from argparse import (
from gettext import gettext
from typing import Dict, List, Set, Tuple
def action_is_open(action):
    """Returns True if action could consume more arguments (i.e., its pattern is open)."""
    num_consumed_args = _num_consumed_args.get(action, 0)
    if action.nargs in [ZERO_OR_MORE, ONE_OR_MORE, PARSER, REMAINDER]:
        return True
    if action.nargs == OPTIONAL or action.nargs is None:
        return num_consumed_args == 0
    assert isinstance(action.nargs, int), 'failed to handle a possible nargs value: %r' % action.nargs
    return num_consumed_args < action.nargs