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
class BaseHdfsAndPosixUrl(StorageUrl):
    """Base class designed for HDFS and POSIX file system URLs.

  Attributes:
    scheme (ProviderPrefix): The cloud provider, must be either POSIX or HDFS.
    bucket_name (str): None.
    object_name (str): The file/directory path.
    generation (str): None.
  """

    def __init__(self, scheme, url_string):
        """Initialize BaseHadoopAndPosixUrl instance."""
        super(BaseHdfsAndPosixUrl, self).__init__()
        self.scheme = scheme
        self.bucket_name = None
        self.generation = None
        self.object_name = url_string[len(scheme.value + SCHEME_DELIMITER):]
        if self.scheme not in [ProviderPrefix.POSIX, ProviderPrefix.HDFS]:
            raise errors.InvalidUrlError('Unrecognized scheme "%s"' % self.scheme)
        if not self.object_name.startswith(self.delimiter):
            log.warning('{} URLs typically start at the root directory. Did you mean: {}{}{}{}'.format(self.scheme.name, self.scheme.value, SCHEME_DELIMITER, self.delimiter, self.object_name))

    @property
    def delimiter(self):
        """Returns the pathname separator character used by POSIX and HDFS."""
        return '/'

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