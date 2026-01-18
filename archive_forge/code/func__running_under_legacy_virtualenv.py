import logging
import os
import re
import site
import sys
from typing import List, Optional
def _running_under_legacy_virtualenv() -> bool:
    """Checks if sys.real_prefix is set.

    This handles virtual environments created with pypa's virtualenv.
    """
    return hasattr(sys, 'real_prefix')