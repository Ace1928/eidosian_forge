from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class BucketListingObject(BucketListingRef):
    """BucketListingRef subclass for objects."""

    def __init__(self, storage_url, root_object=None):
        """Creates a BucketListingRef of type object.

    Args:
      storage_url: StorageUrl containing an object.
      root_object: Underlying object metadata, if available.
    """
        super(BucketListingObject, self).__init__()
        self._ref_type = self._BucketListingRefType.OBJECT
        self._url_string = storage_url.url_string
        self.storage_url = storage_url
        self.root_object = root_object