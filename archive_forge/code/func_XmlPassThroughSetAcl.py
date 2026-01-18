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
def XmlPassThroughSetAcl(self, acl_text, storage_url, canned=True, def_obj_acl=False, provider=None):
    """XML compatibility function for setting ACLs.

    Args:
      acl_text: XML ACL or canned ACL string.
      storage_url: StorageUrl object.
      canned: If true, acl_text is treated as a canned ACL string.
      def_obj_acl: If true, set the default object ACL on a bucket.
      provider: Cloud storage provider to connect to.  If not present,
                class-wide default is used.

    Raises:
      ArgumentException for errors during input validation.
      ServiceException for errors interacting with cloud storage providers.

    Returns:
      None.
    """
    self._GetApi(provider).XmlPassThroughSetAcl(acl_text, storage_url, canned=canned, def_obj_acl=def_obj_acl)