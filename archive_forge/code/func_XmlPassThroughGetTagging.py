from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
from boto import config
from gslib import context_config
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import CloudApi
from gslib.cs_api_map import ApiMapConstants
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
def XmlPassThroughGetTagging(self, storage_url, provider=None):
    """XML compatibility function for getting tagging configuration on a bucket.

    Args:
      storage_url: StorageUrl object.
      provider: Cloud storage provider to connect to.  If not present,
                class-wide default is used.

    Raises:
      ArgumentException for errors during input validation.
      ServiceException for errors interacting with cloud storage providers.

    Returns:
      Tagging configuration XML for the bucket specified by storage_url.
    """
    return self._GetApi(provider).XmlPassThroughGetTagging(storage_url)