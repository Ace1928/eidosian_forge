import codecs
import errno
import os
import random
import shutil
import sys
from typing import Any, Dict
class FormatSafeDict(Dict[Any, Any]):
    """Format a dictionary safely."""

    def __missing__(self, key):
        """Handle missing value."""
        return '{' + key + '}'