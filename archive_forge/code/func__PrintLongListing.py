from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
from gslib.cloud_api import NotFoundException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import ENCRYPTED_FIELDS
from gslib.utils.ls_helper import LsHelper
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.ls_helper import UNENCRYPTED_FULL_LISTING_FIELDS
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils import text_util
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import LabelTranslation
from gslib.utils.unit_util import MakeHumanReadable
def _PrintLongListing(self, bucket_listing_ref):
    """Prints an object with ListingStyle.LONG."""
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
    timestamp = JSON_TIMESTAMP_RE.sub('\\1T\\2Z', str(obj.timeCreated))
    printstr = '%(size)10s  %(timestamp)s  %(url)s'
    encoded_etag = None
    encoded_metagen = None
    if self.all_versions:
        printstr += '  metageneration=%(metageneration)s'
        encoded_metagen = str(obj.metageneration)
    if self.include_etag:
        printstr += '  etag=%(etag)s'
        encoded_etag = obj.etag
    format_args = {'size': size_string, 'timestamp': timestamp, 'url': url_str, 'metageneration': encoded_metagen, 'etag': encoded_etag}
    text_util.print_to_fd(printstr % format_args)
    return (num_objs, num_bytes)