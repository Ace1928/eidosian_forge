import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class OAuth2InvalidClient(OAuth2Error):

    def __init__(self, code, title, message):
        error_title = 'invalid_client'
        super().__init__(code, title, error_title, message)