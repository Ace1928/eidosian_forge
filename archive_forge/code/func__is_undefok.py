from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from absl import flags
def _is_undefok(arg, undefok_names):
    """Returns whether we can ignore arg based on a set of undefok flag names."""
    if not arg.startswith('-'):
        return False
    if arg.startswith('--'):
        arg_without_dash = arg[2:]
    else:
        arg_without_dash = arg[1:]
    if '=' in arg_without_dash:
        name, _ = arg_without_dash.split('=', 1)
    else:
        name = arg_without_dash
    if name in undefok_names:
        return True
    return False