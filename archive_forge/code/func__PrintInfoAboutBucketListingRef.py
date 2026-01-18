from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import locale
import sys
import six
from gslib.bucket_listing_ref import BucketListingObject
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils import ls_helper
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import print_to_fd
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils import text_util
def _PrintInfoAboutBucketListingRef(self, bucket_listing_ref):
    """Print listing info for given bucket_listing_ref.

    Args:
      bucket_listing_ref: BucketListing being listed.

    Returns:
      Tuple (number of objects, object size)

    Raises:
      Exception: if calling bug encountered.
    """
    obj = bucket_listing_ref.root_object
    url_str = bucket_listing_ref.url_string
    if obj.metadata and S3_DELETE_MARKER_GUID in obj.metadata.additionalProperties:
        size_string = '0'
        num_bytes = 0
        num_objs = 0
        url_str += '<DeleteMarker>'
    else:
        size_string = MakeHumanReadable(obj.size) if self.human_readable else str(obj.size)
        num_bytes = obj.size
        num_objs = 1
    if not self.summary_only:
        url_detail = '{size:<11}  {url}{ending}'.format(size=size_string, url=six.ensure_text(url_str), ending=six.ensure_text(self.line_ending))
        print_to_fd(url_detail, file=sys.stdout, end='')
    return (num_objs, num_bytes)