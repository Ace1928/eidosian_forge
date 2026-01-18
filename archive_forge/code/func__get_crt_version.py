import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _get_crt_version():
    """
    This function is considered private and is subject to abrupt breaking
    changes.
    """
    try:
        import awscrt
        return awscrt.__version__
    except AttributeError:
        return None