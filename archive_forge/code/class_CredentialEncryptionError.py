import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class CredentialEncryptionError(Exception):
    message_format = _('An unexpected error prevented the server from accessing encrypted credentials.')