import argparse
import os
from keystoneauth1.loading import _utils
from keystoneauth1.loading import base
from keystoneauth1 import session
def _positive_non_zero_float(argument_value):
    if argument_value is None:
        return None
    try:
        value = float(argument_value)
    except ValueError:
        msg = '%s must be a float' % argument_value
        raise argparse.ArgumentTypeError(msg)
    if value <= 0:
        msg = '%s must be greater than 0' % argument_value
        raise argparse.ArgumentTypeError(msg)
    return value