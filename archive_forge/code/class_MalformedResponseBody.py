from oslo_utils import encodeutils
from neutronclient._i18n import _
class MalformedResponseBody(NeutronClientException):
    message = _('Malformed response body: %(reason)s')