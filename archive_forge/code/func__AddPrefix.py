import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _AddPrefix(element, prefix):
    """Add |prefix| to |element| or each subelement if element is iterable."""
    if element is None:
        return element
    if isinstance(element, list) or isinstance(element, tuple):
        return [prefix + e for e in element]
    else:
        return prefix + element