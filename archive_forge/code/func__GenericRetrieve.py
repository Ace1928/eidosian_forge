import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _GenericRetrieve(root, default, path):
    """Given a list of dictionary keys |path| and a tree of dicts |root|, find
    value at path, or return |default| if any of the path doesn't exist."""
    if not root:
        return default
    if not path:
        return root
    return _GenericRetrieve(root.get(path[0]), default, path[1:])