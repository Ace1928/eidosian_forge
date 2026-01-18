from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import json
import sys
def _verify_scikit_learn(version):
    """Check whether scikit-learn is installed at an appropriate version."""
    try:
        import scipy
    except ImportError:
        eprint('Cannot import scipy, which is needed for scikit-learn. Please verify "python -c \'import scipy\'" works.')
        return False
    try:
        import sklearn
    except ImportError:
        eprint('Cannot import sklearn. Please verify "python -c \'import sklearn\'" works.')
        return False
    try:
        if sklearn.__version__ < version:
            eprint('Scikit-learn version must be at least {} .'.format(version), VERIFY_SCIKIT_LEARN_VERSION)
            return False
    except (NameError, AttributeError) as e:
        eprint('Error while getting the installed scikit-learn version: ', e, '\n', VERIFY_SCIKIT_LEARN_VERSION)
        return False
    return True