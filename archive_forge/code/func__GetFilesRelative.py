from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import zip
def _GetFilesRelative(root):
    """Return all the descendents of root, relative to its path.

  For instance, given the following directory structure

      /path/to/root/a
      /path/to/root/a/b
      /path/to/root/c

  This function would return `['a', 'a/b', 'c']`.

  Args:
    root: str, the path to list descendents of.

  Returns:
    list of str, the paths in the given directory.
  """
    paths = []
    for dirpath, _, filenames in os.walk(six.text_type(root)):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            paths.append(os.path.relpath(abs_path, start=root))
    return paths