import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class MetadataFileError(UnexpectedError):
    debug_message_format = _('Error while reading metadata file: %(reason)s.')