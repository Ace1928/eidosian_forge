from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import json
import sys
def _verify_xgboost(version):
    """Check whether xgboost is installed at an appropriate version."""
    try:
        import xgboost
    except ImportError:
        eprint('Cannot import xgboost. Please verify "python -c \'import xgboost\'" works.')
        return False
    try:
        if xgboost.__version__ < version:
            eprint('Xgboost version must be at least {} .'.format(version), VERIFY_XGBOOST_VERSION)
            return False
    except (NameError, AttributeError) as e:
        eprint('Error while getting the installed xgboost version: ', e, '\n', VERIFY_XGBOOST_VERSION)
        return False
    return True