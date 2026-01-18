import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
class GypError(Exception):
    """Error class representing an error, which is to be presented
  to the user.  The main entry point will catch and display this.
  """
    pass