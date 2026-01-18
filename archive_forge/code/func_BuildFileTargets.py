import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def BuildFileTargets(target_list, build_file):
    """From a target_list, returns the subset from the specified build_file.
  """
    return [p for p in target_list if BuildFile(p) == build_file]