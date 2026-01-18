import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class FederatedProtocolNotFound(NotFound):
    message_format = _('Could not find federated protocol %(protocol_id)s for Identity Provider: %(idp_id)s.')