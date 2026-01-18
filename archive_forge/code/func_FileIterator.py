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
def FileIterator(base, skip_files):
    """Walks a directory tree, returning all the files. Follows symlinks.

  Args:
    base: The base path to search for files under.
    skip_files: A regular expression object for files/directories to skip.

  Yields:
    Paths of files found, relative to base.
  """
    dirs = ['']
    while dirs:
        current_dir = dirs.pop()
        entries = set(os.listdir(os.path.join(base, current_dir)))
        for entry in sorted(entries):
            name = os.path.join(current_dir, entry)
            fullname = os.path.join(base, name)
            if os.path.isfile(fullname):
                if ShouldSkip(skip_files, name):
                    log.info('Ignoring file [%s]: File matches ignore regex.', name)
                else:
                    yield name
            elif os.path.isdir(fullname):
                if ShouldSkip(skip_files, name):
                    log.info('Ignoring directory [%s]: Directory matches ignore regex.', name)
                else:
                    dirs.append(name)