import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def EncodePOSIXShellList(list):
    """Encodes |list| suitably for consumption by POSIX shells.

  Returns EncodePOSIXShellArgument for each item in list, and joins them
  together using the space character as an argument separator.
  """
    encoded_arguments = []
    for argument in list:
        encoded_arguments.append(EncodePOSIXShellArgument(argument))
    return ' '.join(encoded_arguments)