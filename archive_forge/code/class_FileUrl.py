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
class FileUrl(StorageUrl):
    """File URL class providing parsing and convenience methods.

  This class assists with usage and manipulation of an
  (optionally wildcarded) file URL string.  Depending on the string
  contents, this class represents one or more directories or files.

  Attributes:
    scheme (ProviderPrefix): This will always be "file" for FileUrl.
    bucket_name (str): None for FileUrl.
    object_name (str): The file/directory path.
    generation (str): None for FileUrl.
  """

    def __init__(self, url_string):
        """Initialize FileUrl instance.

    Args:
      url_string (str): The string representing the filepath.
    """
        super(FileUrl, self).__init__()
        self.scheme = ProviderPrefix.FILE
        self.bucket_name = None
        self.generation = None
        if url_string.startswith('file://'):
            filename = url_string[len('file://'):]
        else:
            filename = url_string
        if platforms.OperatingSystem.IsWindows():
            self.object_name = filename.replace('/', os.sep)
        else:
            self.object_name = filename
        self._warn_if_unsupported_double_wildcard()

    def _warn_if_unsupported_double_wildcard(self):
        """Log warning if ** use may lead to undefined results."""
        if not self.object_name:
            return
        delimiter_bounded_url = self.delimiter + self.object_name + self.delimiter
        split_url = delimiter_bounded_url.split('{delim}**{delim}'.format(delim=self.delimiter))
        removed_correct_double_wildcards_url_string = ''.join(split_url)
        if '**' in removed_correct_double_wildcards_url_string:
            log.warning('** behavior is undefined if directly preceded or followed by with characters other than / in the cloud and {} locally.'.format(os.sep))

    @property
    def delimiter(self):
        """Returns the pathname separator character used by the OS."""
        return os.sep

    @property
    def is_stream(self):
        """Returns True if the URL points to a named pipe (FIFO) or other stream."""
        return self.is_stdio or is_named_pipe(self.object_name)

    @property
    def is_stdio(self):
        """Returns True if the URL points to stdin or stdout."""
        return self.object_name == '-'

    def exists(self):
        """Returns True if the file/directory exists."""
        return os.path.exists(self.object_name)

    def isdir(self):
        """Returns True if the path represents a directory."""
        return os.path.isdir(self.object_name)

    @property
    def url_string(self):
        """Returns the string representation of the instance."""
        return '{}{}{}'.format(self.scheme.value, SCHEME_DELIMITER, self.object_name)

    @property
    def versionless_url_string(self):
        """Returns the string representation of the instance.

    Same as url_string because these files are not versioned.
    """
        return self.url_string