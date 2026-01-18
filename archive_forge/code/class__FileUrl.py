from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
class _FileUrl(StorageUrl):
    """File URL class providing parsing and convenience methods.

    This class assists with usage and manipulation of an
    (optionally wildcarded) file URL string.  Depending on the string
    contents, this class represents one or more directories or files.

    For File URLs, scheme is always file, bucket_name is always blank,
    and object_name contains the file/directory path.
  """

    def __init__(self, url_string, is_stream=False, is_fifo=False):
        self.scheme = 'file'
        self.delim = os.sep
        self.bucket_name = ''
        match = FILE_OBJECT_REGEX.match(url_string)
        if match and match.lastindex == 2:
            self.object_name = match.group(2)
        else:
            self.object_name = url_string
        if system_util.IS_WINDOWS:
            self.object_name = self.object_name.replace('/', '\\')
        self.generation = None
        self.is_stream = is_stream
        self.is_fifo = is_fifo
        self._WarnIfUnsupportedDoubleWildcard()

    def Clone(self):
        return _FileUrl(self.url_string)

    def IsFileUrl(self):
        return True

    def IsCloudUrl(self):
        return False

    def IsStream(self):
        return self.is_stream

    def IsFifo(self):
        return self.is_fifo

    def IsDirectory(self):
        return not self.IsStream() and (not self.IsFifo()) and os.path.isdir(self.object_name)

    def CreatePrefixUrl(self, wildcard_suffix=None):
        return self.url_string

    @property
    def url_string(self):
        return '%s://%s' % (self.scheme, self.object_name)

    @property
    def versionless_url_string(self):
        return self.url_string

    def __str__(self):
        return self.url_string