from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
def StorageUri(self, uri_str):
    """Instantiates StorageUri using class state and gsutil default flag values.

    Args:
      uri_str: StorageUri naming bucket or object.

    Returns:
      boto.StorageUri for given uri_str.

    Raises:
      InvalidUriError: if uri_str not valid.
    """
    return boto.storage_uri(uri_str, 'file', debug=self.debug, validate=False, bucket_storage_uri_class=self.bucket_storage_uri_class, suppress_consec_slashes=False)