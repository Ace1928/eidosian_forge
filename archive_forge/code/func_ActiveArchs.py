import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def ActiveArchs(self, archs, valid_archs, sdkroot):
    """Expands variables references in ARCHS, and filter by VALID_ARCHS if it
    is defined (if not set, Xcode accept any value in ARCHS, otherwise, only
    values present in VALID_ARCHS are kept)."""
    expanded_archs = self._ExpandArchs(archs or self._default, sdkroot or '')
    if valid_archs:
        filtered_archs = []
        for arch in expanded_archs:
            if arch in valid_archs:
                filtered_archs.append(arch)
        expanded_archs = filtered_archs
    return expanded_archs