import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def _get_or_default(mylist, i, default=None):
    """return list item number, or default if don't exist"""
    if i >= len(mylist):
        return default
    else:
        return mylist[i]