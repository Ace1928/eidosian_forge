from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import glob
import logging
import os
import re
import textwrap
import six
from gslib.bucket_listing_ref import BucketListingBucket
from gslib.bucket_listing_ref import BucketListingObject
from gslib.bucket_listing_ref import BucketListingPrefix
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import NotFoundException
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import StripOneSlash
from gslib.storage_url import WILDCARD_REGEX
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.text_util import FixWindowsEncodingIfNeeded
from gslib.utils.text_util import PrintableStr
class FileWildcardIterator(WildcardIterator):
    """WildcardIterator subclass for files and directories.

  If you use recursive wildcards ('**') only a single such wildcard is
  supported. For example you could use the wildcard '**/*.txt' to list all .txt
  files in any subdirectory of the current directory, but you couldn't use a
  wildcard like '**/abc/**/*.txt' (which would, if supported, let you find .txt
  files in any subdirectory named 'abc').
  """

    def __init__(self, wildcard_url, exclude_tuple=None, ignore_symlinks=False, logger=None):
        """Instantiates an iterator over BucketListingRefs matching wildcard URL.

    Args:
      wildcard_url: FileUrl that contains the wildcard to iterate.
      exclude_tuple: (base_url, exclude_pattern), where base_url is
                     top-level URL to list; exclude_pattern is a regex
                     of paths to ignore during iteration.
      ignore_symlinks: If True, ignore symlinks during iteration.
      logger: logging.Logger used for outputting debug messages during
              iteration. If None, the root logger will be used.
    """
        self.wildcard_url = wildcard_url
        self.exclude_tuple = exclude_tuple
        self.ignore_symlinks = ignore_symlinks
        self.logger = logger or logging.getLogger()

    def __iter__(self, bucket_listing_fields=None):
        """Iterator that gets called when iterating over the file wildcard.

    In the case where no wildcard is present, returns a single matching file
    or directory.

    Args:
      bucket_listing_fields: Iterable fields to include in listings.
          Ex. ['size']. Currently only 'size' is supported.
          If present, will populate yielded BucketListingObject.root_object
          with the file name and size.

    Raises:
      WildcardException: if invalid wildcard found.

    Yields:
      BucketListingRef of type OBJECT (for files) or PREFIX (for directories)
    """
        include_size = bucket_listing_fields and 'size' in set(bucket_listing_fields)
        wildcard = self.wildcard_url.object_name
        match = FLAT_LIST_REGEX.match(wildcard)
        if match:
            base_dir = match.group('before')[:-1]
            remaining_wildcard = match.group('after')
            if remaining_wildcard.startswith('*'):
                raise WildcardException('Invalid wildcard with more than 2 consecutive *s (%s)' % wildcard)
            if not remaining_wildcard:
                remaining_wildcard = '*'
            remaining_wildcard = remaining_wildcard.lstrip(os.sep)
            filepaths = self._IterDir(base_dir, remaining_wildcard)
        else:
            filepaths = glob.iglob(wildcard)
        for filepath in filepaths:
            expanded_url = StorageUrlFromString(filepath)
            try:
                if self.ignore_symlinks and os.path.islink(filepath):
                    if self.logger:
                        self.logger.info('Skipping symbolic link %s...', filepath)
                    continue
                if os.path.isdir(filepath):
                    yield BucketListingPrefix(expanded_url)
                else:
                    blr_object = _GetFileObject(filepath) if include_size else None
                    yield BucketListingObject(expanded_url, root_object=blr_object)
            except UnicodeEncodeError:
                raise CommandException('\n'.join(textwrap.wrap(_UNICODE_EXCEPTION_TEXT % repr(filepath))))

    def _IterDir(self, directory, wildcard):
        """An iterator over the specified dir and wildcard.

    Args:
      directory (unicode): The path of the directory to iterate over.
      wildcard (str): The wildcard characters used for filename pattern
          matching.

    Yields:
      (str) A string containing the path to a file somewhere under the directory
      hierarchy of `directory`.

    Raises:
      ComandException: If this method encounters a file path that it cannot
      decode as UTF-8.
    """
        if os.path.splitdrive(directory)[0] == directory:
            directory += '\\'
        for dirpath, dirnames, filenames in os.walk(six.ensure_text(directory), topdown=True):
            filtered_dirnames = []
            for dirname in dirnames:
                full_dir_path = os.path.join(dirpath, dirname)
                if not self._ExcludeDir(full_dir_path):
                    filtered_dirnames.append(dirname)
                else:
                    continue
                if self.logger and os.path.islink(full_dir_path):
                    self.logger.info('Skipping symlink directory "%s"', full_dir_path)
            dirnames[:] = filtered_dirnames
            for f in fnmatch.filter(filenames, wildcard):
                try:
                    yield os.path.join(dirpath, FixWindowsEncodingIfNeeded(f))
                except UnicodeDecodeError:
                    raise CommandException('\n'.join(textwrap.wrap(_UNICODE_EXCEPTION_TEXT % repr(os.path.join(dirpath, f)))))

    def _ExcludeDir(self, dir):
        """Check a directory to see if it should be excluded from os.walk.

    Args:
      dir: String representing the directory to check.

    Returns:
      True if the directory should be excluded.
    """
        if self.exclude_tuple is None:
            return False
        base_url, exclude_dirs, exclude_pattern = self.exclude_tuple
        if not exclude_dirs:
            return False
        str_to_check = StorageUrlFromString(dir).url_string[len(base_url.url_string):]
        if str_to_check.startswith(self.wildcard_url.delim):
            str_to_check = str_to_check[1:]
        if exclude_pattern.match(str_to_check):
            if self.logger:
                self.logger.info('Skipping excluded directory %s...', dir)
            return True

    def IterObjects(self, bucket_listing_fields=None):
        """Iterates over the wildcard, yielding only object (file) refs.

    Args:
      bucket_listing_fields: Iterable fields to include in listings.
          Ex. ['size']. Currently only 'size' is supported.
          If present, will populate yielded BucketListingObject.root_object
          with the file name and size.

    Yields:
      BucketListingRefs of type OBJECT or empty iterator if no matches.
    """
        for bucket_listing_ref in self.IterAll(bucket_listing_fields=bucket_listing_fields):
            if bucket_listing_ref.IsObject():
                yield bucket_listing_ref

    def IterAll(self, bucket_listing_fields=None, expand_top_level_buckets=False):
        """Iterates over the wildcard, yielding BucketListingRefs.

    Args:
      bucket_listing_fields: Iterable fields to include in listings.
          Ex. ['size']. Currently only 'size' is supported.
          If present, will populate yielded BucketListingObject.root_object
          with the file name and size.
      expand_top_level_buckets: Ignored; filesystems don't have buckets.

    Yields:
      BucketListingRefs of type OBJECT (file) or PREFIX (directory),
      or empty iterator if no matches.
    """
        for bucket_listing_ref in self.__iter__(bucket_listing_fields=bucket_listing_fields):
            yield bucket_listing_ref

    def IterBuckets(self, unused_bucket_fields=None):
        """Placeholder to allow polymorphic use of WildcardIterator.

    Args:
      unused_bucket_fields: Ignored; filesystems don't have buckets.

    Raises:
      WildcardException: in all cases.
    """
        raise WildcardException('Iterating over Buckets not possible for file wildcards')