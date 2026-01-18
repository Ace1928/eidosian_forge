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
def XmlPassThroughGetAcl(self, storage_url, def_obj_acl=False, provider=None):
    """XML compatibility function for getting ACLs.

    Args:
      storage_url: StorageUrl object.
      def_obj_acl: If true, get the default object ACL on a bucket.
      provider: Cloud storage provider to connect to.  If not present,
                class-wide default is used.

    Raises:
      ArgumentException for errors during input validation.
      ServiceException for errors interacting with cloud storage providers.

    Returns:
      ACL XML for the resource specified by storage_url.
    """
    return self._GetApi(provider).XmlPassThroughGetAcl(storage_url, def_obj_acl=def_obj_acl)