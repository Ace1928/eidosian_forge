import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class OAuth2UnsupportedGrantType(OAuth2Error):

    def __init__(self, code, title, message):
        error_title = 'unsupported_grant_type'
        super().__init__(code, title, error_title, message)