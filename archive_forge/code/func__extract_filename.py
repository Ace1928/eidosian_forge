from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _extract_filename(self, flagfile_str):
    """Returns filename from a flagfile_str of form -[-]flagfile=filename.

    The cases of --flagfile foo and -flagfile foo shouldn't be hitting
    this function, as they are dealt with in the level above this
    function.

    Args:
      flagfile_str: str, the flagfile string.

    Returns:
      str, the filename from a flagfile_str of form -[-]flagfile=filename.

    Raises:
      Error: Raised when illegal --flagfile is provided.
    """
    if flagfile_str.startswith('--flagfile='):
        return os.path.expanduser(flagfile_str[len('--flagfile='):].strip())
    elif flagfile_str.startswith('-flagfile='):
        return os.path.expanduser(flagfile_str[len('-flagfile='):].strip())
    else:
        raise _exceptions.Error('Hit illegal --flagfile type: %s' % flagfile_str)