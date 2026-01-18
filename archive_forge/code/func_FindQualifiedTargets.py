import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def FindQualifiedTargets(target, qualified_list):
    """
  Given a list of qualified targets, return the qualified targets for the
  specified |target|.
  """
    return [t for t in qualified_list if ParseQualifiedTarget(t)[1] == target]