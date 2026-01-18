import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _AppendOrReturn(append, element):
    """If |append| is None, simply return |element|. If |append| is not None,
    then add |element| to it, adding each item in |element| if it's a list or
    tuple."""
    if append is not None and element is not None:
        if isinstance(element, list) or isinstance(element, tuple):
            append.extend(element)
        else:
            append.append(element)
    else:
        return element