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
class StorageUrl(object):
    """Abstract base class for file and Cloud Storage URLs."""

    def Clone(self):
        raise NotImplementedError('Clone not overridden')

    def IsFileUrl(self):
        raise NotImplementedError('IsFileUrl not overridden')

    def IsCloudUrl(self):
        raise NotImplementedError('IsCloudUrl not overridden')

    def IsStream():
        raise NotImplementedError('IsStream not overridden')

    def IsFifo(self):
        raise NotImplementedError('IsFifo not overridden')

    def CreatePrefixUrl(self, wildcard_suffix=None):
        """Returns a prefix of this URL that can be used for iterating.

    Args:
      wildcard_suffix: If supplied, this wildcard suffix will be appended to the
                       prefix with a trailing slash before being returned.

    Returns:
      A prefix of this URL that can be used for iterating.

    If this URL contains a trailing slash, it will be stripped to create the
    prefix. This helps avoid infinite looping when prefixes are iterated, but
    preserves other slashes so that objects with '/' in the name are handled
    properly.

    For example, when recursively listing a bucket with the following contents:
      gs://bucket// <-- object named slash
      gs://bucket//one-dir-deep
    a top-level expansion with '/' as a delimiter will result in the following
    URL strings:
      'gs://bucket//' : OBJECT
      'gs://bucket//' : PREFIX
    If we right-strip all slashes from the prefix entry and add a wildcard
    suffix, we will get 'gs://bucket/*' which will produce identical results
    (and infinitely recurse).

    Example return values:
      ('gs://bucket/subdir/', '*') becomes 'gs://bucket/subdir/*'
      ('gs://bucket/', '*') becomes 'gs://bucket/*'
      ('gs://bucket/', None) becomes 'gs://bucket'
      ('gs://bucket/subdir//', '*') becomes 'gs://bucket/subdir//*'
      ('gs://bucket/subdir///', '**') becomes 'gs://bucket/subdir///**'
      ('gs://bucket/subdir/', '*') where 'subdir/' is an object becomes
           'gs://bucket/subdir/*', but iterating on this will return 'subdir/'
           as a BucketListingObject, so we will not recurse on it as a subdir
           during listing.
    """
        raise NotImplementedError('CreatePrefixUrl not overridden')

    def _WarnIfUnsupportedDoubleWildcard(self):
        """Warn if ** use may lead to undefined results."""
        if not self.object_name:
            return
        delimiter_bounded_url = self.delim + self.object_name + self.delim
        split_url = delimiter_bounded_url.split('{delim}**{delim}'.format(delim=self.delim))
        removed_correct_double_wildcards_url_string = ''.join(split_url)
        if '**' in removed_correct_double_wildcards_url_string:
            sys.stderr.write('** behavior is undefined if directly preceeded or followed by with characters other than / in the cloud and {} locally.'.format(os.sep))

    @property
    def url_string(self):
        raise NotImplementedError('url_string not overridden')

    @property
    def versionless_url_string(self):
        raise NotImplementedError('versionless_url_string not overridden')

    def __eq__(self, other):
        return isinstance(other, StorageUrl) and self.url_string == other.url_string

    def __hash__(self):
        return hash(self.url_string)