from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
def ShouldSkip(skip_files, path):
    """Returns whether the given path should be skipped by the skip_files field.

  A user can specify a `skip_files` field in their .yaml file, which is a list
  of regular expressions matching files that should be skipped. By this point in
  the code, it's been turned into one mega-regex that matches any file to skip.

  Args:
    skip_files: A regular expression object for files/directories to skip.
    path: str, the path to the file/directory which might be skipped (relative
      to the application root)

  Returns:
    bool, whether the file/dir should be skipped.
  """
    path = ConvertToPosixPath(path)
    return skip_files.match(path)