from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.ext import builtins
def _ResolvePath(included_from, included_path, basepath):
    """Gets the absolute path of the file to be included.

  Resolves in the following order:
  - absolute path or relative to working directory
    (path as specified resolves to a file)
  - relative to basepath
    (basepath + path resolves to a file)
  - relative to file it was included from
    (included_from + included_path resolves to a file)

  Args:
    included_from: absolute path of file that included_path was included from.
    included_path: file string from includes directive.
    basepath: the application directory.

  Returns:
    absolute path of the first file found for included_path or ''.
  """
    path = os.path.join(os.path.dirname(included_from), included_path)
    if not _IsFileOrDirWithFile(path):
        path = os.path.join(basepath, included_path)
        if not _IsFileOrDirWithFile(path):
            path = included_path
            if not _IsFileOrDirWithFile(path):
                return ''
    if os.path.isfile(path):
        return os.path.normcase(os.path.abspath(path))
    return os.path.normcase(os.path.abspath(os.path.join(path, 'include.yaml')))