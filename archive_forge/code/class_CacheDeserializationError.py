import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class CacheDeserializationError(Exception):

    def __init__(self, obj, data):
        super(CacheDeserializationError, self).__init__(_('Failed to deserialize %(obj)s. Data is %(data)s') % {'obj': obj, 'data': data})