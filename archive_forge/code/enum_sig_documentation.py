from __future__ import print_function, absolute_import
import sys
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature

    HintingEnumerator enumerates all signatures in a module slightly changed.

    This class is used for generating complete listings of all signatures for
    hinting stubs. Only default values are replaced by "...".
    