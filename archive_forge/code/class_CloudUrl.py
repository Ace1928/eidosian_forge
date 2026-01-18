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
class CloudUrl(StorageUrl):
    """Cloud URL class providing parsing and convenience methods.

    This class assists with usage and manipulation of an
    (optionally wildcarded) cloud URL string.  Depending on the string
    contents, this class represents a provider, bucket(s), or object(s).

    This class operates only on strings.  No cloud storage API calls are
    made from this class.

    Attributes:
      scheme (ProviderPrefix): The cloud provider.
      bucket_name (str|None): The bucket name if url represents an object or
        bucket.
      object_name (str|None): The object name if url represents an object or
        prefix.
      generation (str|None): The generation number if present.
  """
    CLOUD_URL_DELIM = '/'

    def __init__(self, scheme, bucket_name=None, object_name=None, generation=None):
        super(CloudUrl, self).__init__()
        self.scheme = scheme if scheme else None
        self.bucket_name = bucket_name if bucket_name else None
        self.object_name = object_name if object_name else None
        self.generation = str(generation) if generation else None
        self._validate_scheme()
        self._validate_object_name()

    @classmethod
    def from_url_string(cls, url_string):
        """Parse the url string and return the storage url object.

    Args:
      url_string (str): Cloud storage url of the form gs://bucket/object

    Returns:
      CloudUrl object

    Raises:
      InvalidUrlError: Raised if the url_string is not a valid cloud url.
    """
        scheme = _get_scheme_from_url_string(url_string)
        schemeless_url_string = url_string[len(scheme.value + SCHEME_DELIMITER):]
        if schemeless_url_string.startswith('/'):
            raise errors.InvalidUrlError('Cloud URL scheme should be followed by colon and two slashes: "{}". Found: "{}"'.format(SCHEME_DELIMITER, url_string))
        bucket_name, _, object_name = schemeless_url_string.partition(CLOUD_URL_DELIMITER)
        object_name, generation = get_generation_number_from_object_name(scheme, object_name)
        return cls(scheme, bucket_name, object_name, generation)

    def _validate_scheme(self):
        if self.scheme not in VALID_CLOUD_SCHEMES:
            raise errors.InvalidUrlError('Unrecognized scheme "%s"' % self.scheme)

    def _validate_object_name(self):
        if self.object_name == '.' or self.object_name == '..':
            raise errors.InvalidUrlError('%s is an invalid root-level object name.' % self.object_name)

    @property
    def is_stream(self):
        """Cloud URLs cannot represent named pipes (FIFO) or other streams."""
        return False

    @property
    def is_stdio(self):
        """Cloud URLs cannot represent stdin or stdout."""
        return False

    @property
    def url_string(self):
        url_str = self.versionless_url_string
        if self.generation:
            url_str += '#%s' % self.generation
        return url_str

    @property
    def versionless_url_string(self):
        if self.is_provider():
            return '{}{}'.format(self.scheme.value, SCHEME_DELIMITER)
        elif self.is_bucket():
            return '{}{}{}/'.format(self.scheme.value, SCHEME_DELIMITER, self.bucket_name)
        return '{}{}{}/{}'.format(self.scheme.value, SCHEME_DELIMITER, self.bucket_name, self.object_name)

    @property
    def delimiter(self):
        return self.CLOUD_URL_DELIM

    def is_bucket(self):
        return bool(self.bucket_name and (not self.object_name))

    def is_object(self):
        return bool(self.bucket_name and self.object_name)

    def is_provider(self):
        return bool(self.scheme and (not self.bucket_name))