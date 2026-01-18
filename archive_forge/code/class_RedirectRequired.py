import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class RedirectRequired(Exception):
    """Error class for redirection.

    Child classes should define an HTTP redirect url
    message_format.

    """
    redirect_url = None
    code = http.client.FOUND

    def __init__(self, redirect_url, **kwargs):
        self.redirect_url = redirect_url
        super(RedirectRequired, self).__init__(**kwargs)