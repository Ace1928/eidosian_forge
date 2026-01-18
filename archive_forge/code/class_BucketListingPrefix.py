from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class BucketListingPrefix(BucketListingRef):
    """BucketListingRef subclass for prefixes."""

    def __init__(self, storage_url, root_object=None):
        """Creates a BucketListingRef of type prefix.

    Args:
      storage_url: StorageUrl containing a prefix.
      root_object: Underlying object metadata, if available.
    """
        super(BucketListingPrefix, self).__init__()
        self._ref_type = self._BucketListingRefType.PREFIX
        self._url_string = storage_url.url_string
        self.storage_url = storage_url
        self.root_object = root_object