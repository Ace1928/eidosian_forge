import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
def _ssl_cert_req_type(self, req_type):
    try:
        import ssl
    except ImportError:
        raise exception.ConfigurationError(_('no ssl support available'))
    req_type = req_type.upper()
    try:
        return {'NONE': ssl.CERT_NONE, 'OPTIONAL': ssl.CERT_OPTIONAL, 'REQUIRED': ssl.CERT_REQUIRED}[req_type]
    except KeyError:
        msg = _('Invalid ssl_cert_reqs value of %s, must be one of "NONE", "OPTIONAL", "REQUIRED"') % req_type
        raise exception.ConfigurationError(msg)