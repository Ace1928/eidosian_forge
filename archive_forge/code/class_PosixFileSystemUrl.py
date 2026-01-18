from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
class PosixFileSystemUrl(BaseHdfsAndPosixUrl):
    """URL class representing local and external POSIX file systems.

  *Intended for transfer component.*

  This class is different from FileUrl in many ways:
  1) It supports only POSIX file systems (not Windows).
  2) It can represent file systems on external machines.
  3) It cannot run checks on the address of the URL like "exists" or "is_stream"
     because the URL may point to a different machine.
  4) The class is intended for use in "agent transfers". This is when a
     Transfer Service customer installs agents on one machine or multiple and
     uses the agent software to upload and download files on the machine(s).

  We implement this class in the "storage" component for convenience and
  because the "storage" and "transfer" products are tightly coupled.

  Attributes:
    scheme (ProviderPrefix): This will always be "posix" for PosixFileSystemUrl.
    bucket_name (None): N/A
    object_name (str): The file/directory path.
    generation (None): N/A
  """

    def __init__(self, url_string):
        """Initialize PosixFileSystemUrl instance.

    Args:
      url_string (str): Local or external POSIX file path.
    """
        super(PosixFileSystemUrl, self).__init__(ProviderPrefix.POSIX, url_string)