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
def ConvertToPosixPath(path):
    """Converts a native-OS path to /-separated: os.path.join('a', 'b')->'a/b'."""
    return posixpath.join(*path.split(os.path.sep))